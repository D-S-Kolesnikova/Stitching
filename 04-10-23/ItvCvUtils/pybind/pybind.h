#pragma once

#include <pybind11/pybind11.h>
#include <iostream>
#include <ccomplex>
#include <cstring>
#include <atomic>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>

namespace py = pybind11;

void waits() {}

//https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html
//
// TODO???: decrement some more refences?

class TextIOWrapper : public std::ostream
{
    template<class TItem>
    class Queue
    {
    public:
        Queue() = default;
        Queue(const Queue&) = delete;
        Queue& operator=(const Queue&) = delete;

        void put(TItem item)
        {
            {
                const std::lock_guard<std::mutex> queue_lock(m_queue_mutex);
                m_queue.emplace(item);
            }
            m_cv.notify_all();
        }

        TItem wait_for_msg_and_pop()
        {
            std::unique_lock<std::mutex> queue_lock(m_queue_mutex);
            if (!m_queue.empty())
            {
                return unsafe_pop();
            }

            std::unique_lock<std::mutex> cv_lock(m_cv_mutex);
            queue_lock.unlock();
            m_cv.wait(cv_lock);

            queue_lock.lock();
            return unsafe_pop();
        }

    protected:
        std::string unsafe_pop()
        {
            TItem item(std::move(m_queue.front()));
            m_queue.pop();
            return item;
        }

    private:
        std::queue<TItem> m_queue;
        std::mutex m_queue_mutex;
        std::condition_variable m_cv;
        std::mutex m_cv_mutex;
    };

    //

    class StreamBuffer : public std::streambuf
    {
    public:
        explicit StreamBuffer(PyObject* sourcePtr)
            : m_sourcePtr(sourcePtr)
            , m_disposed(false)
            , m_creationThreadId(std::this_thread::get_id())
        {
            Py_XINCREF(m_sourcePtr);
            // And finally, start a polling thread:
            m_msgPoppingThread = std::thread(&StreamBuffer::InfinteMsgPopping, this);
        }

        ~StreamBuffer()
        {
            {
                std::lock_guard<std::mutex> _(m_dispose_mutex);
                m_disposed = true;
            }
            m_msgQueue.put(std::string());  // Force unblock `m_msgQueue` waiting in `m_msgPoppingThread`.
            m_msgPoppingThread.join();
            Py_XDECREF(m_sourcePtr);
        }

        std::streamsize xsputn(const char_type* s, std::streamsize n)
        {
            return Writen(s, n);
        }

        int_type overflow(int_type c)
        {
            Writen(reinterpret_cast<char*>(&c), 1);
            // TODO???: std::streambuf::overflow(c);
            return !TextIOWrapper::traits_type::eof();
        }

    private:
        size_t Writen(const char* str, size_t n)
        {
            if (std::this_thread::get_id() == m_creationThreadId)
            {
                return UnsafeWriten(str, n);
            }
            else
            {
                m_msgQueue.put(str);
                return n;  // This isn't true, of course....
            }
        }

        size_t UnsafeWriten(const char* str, size_t n) const
        {
            size_t written(0);
            if (m_sourcePtr)
            {
                auto written_pobj = PyObject_CallMethodOneArg(
                    m_sourcePtr,
                    PyUnicode_FromString("write"),
                    PyUnicode_FromStringAndSize(str, n));
                // Refs counter will be decreased on returned object destruction:
                written = pybind11::reinterpret_borrow<pybind11::int_>(pybind11::handle(written_pobj));
            }
            return written;
        }

        void InfinteMsgPopping()
        {
            while (true)
            {
                auto msg = m_msgQueue.wait_for_msg_and_pop();
                {
                    std::lock_guard<std::mutex> _(m_dispose_mutex);
                    if (m_disposed)
                        return;
                    auto state = PyGILState_Ensure();
                    UnsafeWriten(msg.c_str(), msg.length());
                    PyGILState_Release(state);
                }
            }
        }

    //private:
    public:
        PyObject* const m_sourcePtr;
        std::atomic_bool m_disposed;
        std::mutex m_dispose_mutex;
        std::thread::id const m_creationThreadId;
        std::thread m_msgPoppingThread;
        Queue<std::string> m_msgQueue;
    };

    //

public:
    TextIOWrapper()
        : std::ostream(nullptr)
        , std::ios(0)
    {}

    TextIOWrapper(PyObject* source)
        : TextIOWrapper()
    {
        m_sb = std::make_shared<StreamBuffer>(source);
        rdbuf(m_sb.get());
    }

    TextIOWrapper(const TextIOWrapper& other)
        : std::ostream(other.m_sb.get())
        , m_sb(other.m_sb)
    {}

    TextIOWrapper& operator=(const TextIOWrapper& other)
    {
        m_sb = other.m_sb;
        rdbuf(m_sb.get());
        return *this;
    }

private:
    std::shared_ptr<StreamBuffer> m_sb;
};


PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

template <>
class type_caster<TextIOWrapper>
{
private:
    static constexpr char* TextIOBase_Class_Name = "_io._TextIOBase";

public:
    // Python to C++
    bool load(py::handle src, bool convert)
    {
        if (!src)
        {
            return false;
        }

        PyObject* source = src.ptr();
        /*std::cout
            << "source.ob_refcnt = " << source->ob_refcnt
            << "; source->ob_type->tp_name = " << source->ob_type->tp_name
            << "; source = " << source << std::endl;*/

        // Refs counter will be decreased on returned object destruction:
        const auto mro = py::reinterpret_borrow<py::tuple>(py::handle(source->ob_type->tp_mro));

        auto baseClass_h = mro.begin();
        for (; baseClass_h != mro.end(); ++baseClass_h)
        {
            const auto& baseClassName = reinterpret_cast<PyTypeObject*>(baseClass_h->ptr())->tp_name;
            if (!strcmp(baseClassName, TextIOBase_Class_Name))
            {
                break;
            }
        }
        if (baseClass_h == mro.end())
        {
            //return false;
        }

        /* Example:
        if (!convert && !PyComplex_Check(src.ptr())) {
            return false;
        }
        Py_complex result = PyComplex_AsCComplex(src.ptr());
        if (result.real == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        value = std::complex<T>((T) result.real, (T) result.imag);*/

        value = TextIOWrapper(source);
        return true;
    }

    // C++ to Python
    static py::handle
    cast(const TextIOWrapper &src, py::return_value_policy policy, py::handle parent) {
        // TODO: here
        //Example: return PyComplex_FromDoubles((double) src.real(), (double) src.imag());
        return py::handle();
    }

    PYBIND11_TYPE_CASTER(TextIOWrapper, const_name("TextIOWrapper"));
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)