#include <ItvCvUtils/Cpuid.h>

#if defined(_WIN32) || defined(__x86_64)

#include <vector>
#include <bitset>
#include <array>
#include <cstring>
#include <mutex>
#include <regex>

#ifdef _WIN32
#include <intrin.h>
#elif __x86_64__
#include <cpuid.h>
#endif

namespace
{
void cpuid(int registers[4], int InfoType)
{
#ifdef _WIN32
    __cpuidex(registers, InfoType, 0);
#elif __x86_64__
    __cpuid_count(InfoType, 0, registers[0], registers[1], registers[2], registers[3]);
#endif
}
}

namespace ItvCvUtils
{
class Cpuid : public ICpuid
{
private:
    class InstructionSet_Internal
    {
    public:
        InstructionSet_Internal()
        {
            std::array<int, 4> regs;

            // Calling __cpuid with 0x0 as the function_id argument
            // gets the number of the highest valid function ID.
            //__cpuid(regs.data(), 0, regs[0], regs[1],regs[2],regs[3]);
            cpuid(regs.data(), 0);
            m_nIds = regs[0];

            for (int i = 0; i <= m_nIds; ++i)
            {
                cpuid(regs.data(), i);
                m_data.push_back(regs);
            }

            // Capture vendor string
            char vendor[0x20];
            memset(vendor, 0, sizeof(vendor));
            *reinterpret_cast<int *>(vendor) = m_data[0][1];
            *reinterpret_cast<int *>(vendor + 4) = m_data[0][3];
            *reinterpret_cast<int *>(vendor + 8) = m_data[0][2];
            m_vendor = vendor;
            if (m_vendor == "GenuineIntel")
            {
                m_isIntel = true;
            }
            else if (m_vendor == "AuthenticAMD")
            {
                m_isAMD = true;
            }

            // load bitset with flags for function 0x00000001
            if (m_nIds >= 1)
            {
                m_f1ECX = m_data[1][2];
                m_f1EDX = m_data[1][3];
                m_family = ((m_data[1][0] & 0xF00) >> 8);

                int extendedFamily = ((m_data[1][0] & 0xFF00000) >> 20);
                if (m_family == 15)
                    m_family += extendedFamily;

                m_model = ((m_data[1][0] & 0xF0) >> 4);
                int extendedModel = ((m_data[1][0] & 0xF0000) >> 16);
                if (m_model == 15)
                    m_model += extendedModel;

                m_processorType = ((m_data[1][0] & 0x3000) >> 12);
                m_steppingID = (m_data[1][0] & 0xF);
            }

            // load bitset with flags for function 0x00000007
            if (m_nIds >= 7)
            {
                m_f7EBX = m_data[7][1];
                m_f7ECX = m_data[7][2];
            }

            // Calling __cpuid with 0x80000000 as the function_id argument
            // gets the number of the highest valid extended ID.
            cpuid(regs.data(), 0x80000000);
            m_nExIds = regs[0];

            char brand[0x40];
            memset(brand, 0, sizeof(brand));

            for (int i = 0x80000000; i <= m_nExIds; ++i)
            {
                cpuid(regs.data(), i);
                m_extdata.push_back(regs);
            }

            // load bitset with flags for function 0x80000001
            if (m_nExIds >= 0x80000001)
            {
                m_f81ECX = m_extdata[1][2];
                m_f81EDX = m_extdata[1][3];
            }

            // Interpret CPU brand string if reported
            if (m_nExIds >= 0x80000004)
            {
                memcpy(brand, m_extdata[2].data(), sizeof(regs));
                memcpy(brand + 16, m_extdata[3].data(), sizeof(regs));
                memcpy(brand + 32, m_extdata[4].data(), sizeof(regs));
                m_brand = brand;
            }
        };

        int m_nIds = 0;
        int m_nExIds = 0;
        int m_family = 0;
        int m_processorType = 0;
        int m_model = 0;
        int m_steppingID = 0;

        std::string m_vendor;
        std::string m_brand;
        bool m_isIntel = false;
        bool m_isAMD = false;
        std::bitset<32> m_f1ECX;
        std::bitset<32> m_f1EDX;
        std::bitset<32> m_f7EBX;
        std::bitset<32> m_f7ECX;
        std::bitset<32> m_f81ECX;
        std::bitset<32> m_f81EDX;
        std::vector<std::array<int, 4>> m_data;
        std::vector<std::array<int, 4>> m_extdata;

    } m_cpuidData;

public:
    std::string Vendor(void) const override { return m_cpuidData.m_vendor; }
    std::string Brand(void) const override { return m_cpuidData.m_brand; }
    int Family(void) const override {return m_cpuidData.m_family; }
    int Model(void) const override {return m_cpuidData.m_model; }
    int ProcessorType(void) const override {return m_cpuidData.m_processorType; }
    int SteppingID(void) const override {return m_cpuidData.m_steppingID; }

    bool SSE3(void) const override { return m_cpuidData.m_f1ECX[0]; }
    bool PCLMULQDQ(void) const override { return m_cpuidData.m_f1ECX[1]; }
    bool MONITOR(void) const override { return m_cpuidData.m_f1ECX[3]; }
    bool SSSE3(void) const override { return m_cpuidData.m_f1ECX[9]; }
    bool FMA(void) const override { return m_cpuidData.m_f1ECX[12]; }
    bool CMPXCHG16B(void) const override { return m_cpuidData.m_f1ECX[13]; }
    bool SSE41(void) const override { return m_cpuidData.m_f1ECX[19]; }
    bool SSE42(void) const override { return m_cpuidData.m_f1ECX[20]; }
    bool MOVBE(void) const override { return m_cpuidData.m_f1ECX[22]; }
    bool POPCNT(void) const override { return m_cpuidData.m_f1ECX[23]; }
    bool AES(void) const override { return m_cpuidData.m_f1ECX[25]; }
    bool XSAVE(void) const override { return m_cpuidData.m_f1ECX[26]; }
    bool OSXSAVE(void) const override { return m_cpuidData.m_f1ECX[27]; }
    bool AVX(void) const override { return m_cpuidData.m_f1ECX[28]; }
    bool F16C(void) const override { return m_cpuidData.m_f1ECX[29]; }
    bool RDRAND(void) const override { return m_cpuidData.m_f1ECX[30]; }

    bool MSR(void) const override { return m_cpuidData.m_f1EDX[5]; }
    bool CX8(void) const override { return m_cpuidData.m_f1EDX[8]; }
    bool SEP(void) const override { return m_cpuidData.m_f1EDX[11]; }
    bool CMOV(void) const override { return m_cpuidData.m_f1EDX[15]; }
    bool CLFSH(void) const override { return m_cpuidData.m_f1EDX[19]; }
    bool MMX(void) const override { return m_cpuidData.m_f1EDX[23]; }
    bool FXSR(void) const override { return m_cpuidData.m_f1EDX[24]; }
    bool SSE(void) const override { return m_cpuidData.m_f1EDX[25]; }
    bool SSE2(void) const override { return m_cpuidData.m_f1EDX[26]; }

    bool FSGSBASE(void) const override { return m_cpuidData.m_f7EBX[0]; }
    bool BMI1(void) const override { return m_cpuidData.m_f7EBX[3]; }
    bool HLE(void) const override { return m_cpuidData.m_isIntel && m_cpuidData.m_f7EBX[4]; }
    bool AVX2(void) const override { return m_cpuidData.m_f7EBX[5]; }
    bool BMI2(void) const override { return m_cpuidData.m_f7EBX[8]; }
    bool ERMS(void) const override { return m_cpuidData.m_f7EBX[9]; }
    bool INVPCID(void) const override { return m_cpuidData.m_f7EBX[10]; }
    bool RTM(void) const override { return m_cpuidData.m_isIntel && m_cpuidData.m_f7EBX[11]; }
    bool AVX512F(void) const override { return m_cpuidData.m_f7EBX[16]; }
    bool RDSEED(void) const override { return m_cpuidData.m_f7EBX[18]; }
    bool ADX(void) const override { return m_cpuidData.m_f7EBX[19]; }
    bool AVX512PF(void) const override { return m_cpuidData.m_f7EBX[26]; }
    bool AVX512ER(void) const override { return m_cpuidData.m_f7EBX[27]; }
    bool AVX512CD(void) const override { return m_cpuidData.m_f7EBX[28]; }
    bool SHA(void) const override { return m_cpuidData.m_f7EBX[29]; }

    bool PREFETCHWT1(void) const override { return m_cpuidData.m_f7ECX[0]; }

    bool LAHF(void) const override { return m_cpuidData.m_f81ECX[0]; }
    bool LZCNT(void) const override { return m_cpuidData.m_isIntel && m_cpuidData.m_f81ECX[5]; }
    bool ABM(void) const override { return m_cpuidData.m_isAMD && m_cpuidData.m_f81ECX[5]; }
    bool SSE4a(void) const override { return m_cpuidData.m_isAMD && m_cpuidData.m_f81ECX[6]; }
    bool XOP(void) const override { return m_cpuidData.m_isAMD && m_cpuidData.m_f81ECX[11]; }
    bool TBM(void) const override { return m_cpuidData.m_isAMD && m_cpuidData.m_f81ECX[21]; }

    bool SYSCALL(void) const override { return m_cpuidData.m_isIntel && m_cpuidData.m_f81EDX[11]; }
    bool MMXEXT(void) const override { return m_cpuidData.m_isAMD && m_cpuidData.m_f81EDX[22]; }
    bool RDTSCP(void) const override { return m_cpuidData.m_isIntel && m_cpuidData.m_f81EDX[27]; }
    bool _3DNOWEXT(void) const override { return m_cpuidData.m_isAMD && m_cpuidData.m_f81EDX[30]; }
    bool _3DNOW(void) const override { return m_cpuidData.m_isAMD && m_cpuidData.m_f81EDX[31]; }
};

std::shared_ptr<ICpuid> GetCpuid()
{
    static std::weak_ptr<Cpuid> cpuidWp;
    static std::mutex mutexWp;

    std::lock_guard<std::mutex> guard(mutexWp);

    auto cpuidSp = cpuidWp.lock();
    if (!cpuidSp)
    {
        cpuidSp = std::make_shared<Cpuid>();
        cpuidWp = cpuidSp;
    }
    return cpuidSp;
}
}
#else
namespace ItvCvUtils
{
    std::shared_ptr<ICpuid> GetCpuid()
    {
        return nullptr;
    }
}
#endif
