#pragma once

#include <ItvCvUtils/ItvCvUtils.h>

#include <string>
#include <memory>

namespace ItvCvUtils {

class ITVCV_UTILS_API ICpuid
{
public:
    virtual ~ICpuid() = default;

    virtual std::string Vendor(void) const = 0;
    virtual std::string Brand(void) const = 0;
    virtual int Family(void) const = 0; 
    virtual int Model(void) const = 0;
    virtual int ProcessorType(void) const = 0;
    virtual int SteppingID(void) const = 0;

    virtual bool SSE3(void) const = 0;
    virtual bool PCLMULQDQ(void) const = 0;
    virtual bool MONITOR(void) const = 0;
    virtual bool SSSE3(void) const = 0;
    virtual bool FMA(void) const = 0;
    virtual bool CMPXCHG16B(void) const = 0;
    virtual bool SSE41(void) const = 0;
    virtual bool SSE42(void) const = 0;
    virtual bool MOVBE(void) const = 0; 
    virtual bool POPCNT(void) const = 0;
    virtual bool AES(void) const = 0; 
    virtual bool XSAVE(void) const = 0;
    virtual bool OSXSAVE(void) const = 0;
    virtual bool AVX(void) const = 0;
    virtual bool F16C(void) const = 0; 
    virtual bool RDRAND(void) const = 0;

    virtual bool MSR(void) const = 0;
    virtual bool CX8(void) const = 0;
    virtual bool SEP(void) const = 0;
    virtual bool CMOV(void) const = 0;
    virtual bool CLFSH(void) const = 0;
    virtual bool MMX(void) const = 0;
    virtual bool FXSR(void) const = 0;
    virtual bool SSE(void) const = 0;
    virtual bool SSE2(void) const = 0;

    virtual bool FSGSBASE(void) const = 0;
    virtual bool BMI1(void) const = 0;
    virtual bool HLE(void) const = 0;
    virtual bool AVX2(void) const = 0;
    virtual bool BMI2(void) const = 0;
    virtual bool ERMS(void) const = 0;
    virtual bool INVPCID(void) const = 0;
    virtual bool RTM(void) const = 0;
    virtual bool AVX512F(void) const = 0;
    virtual bool RDSEED(void) const = 0;
    virtual bool ADX(void) const = 0;
    virtual bool AVX512PF(void) const = 0;
    virtual bool AVX512ER(void) const = 0;
    virtual bool AVX512CD(void) const = 0;
    virtual bool SHA(void) const = 0;

    virtual bool PREFETCHWT1(void) const = 0;

    virtual bool LAHF(void) const = 0; 
    virtual bool LZCNT(void) const = 0;
    virtual bool ABM(void) const = 0;
    virtual bool SSE4a(void) const = 0;
    virtual bool XOP(void) const = 0;
    virtual bool TBM(void) const = 0;

    virtual bool SYSCALL(void) const = 0;
    virtual bool MMXEXT(void) const = 0;
    virtual bool RDTSCP(void) const = 0;
    virtual bool _3DNOWEXT(void) const = 0;
    virtual bool _3DNOW(void) const = 0;
};

ITVCV_UTILS_API std::shared_ptr<ICpuid> GetCpuid();

}