import onnx
import onnxruntime



def TestNetworkInformation():
    import NetworkInformationPyBind as NI
    import ItvCvUtilsPyBind as CvUtils
    at1 = NI.AttributeSubtype(CvUtils.AttributeAge.Old)
    at2 = NI.AttributeSubtype(CvUtils.AttributeClothes.Boots)
    at3 = NI.AttributeSubtype(CvUtils.AttributeActions.HoldObject)
    at4 = NI.AttributeSubtype(CvUtils.AttributeGender.Male)
    at5 = NI.AttributeSubtype(CvUtils.AttributeOrientation.Back)
    print(at1.WichAttributeType(), at1.GetAgeAttribute())
    print(at2.WichAttributeType(), at2.GetClothesAttribute())
    print(at3.WichAttributeType(), at3.GetActionAttribute())
    print(at4.WichAttributeType(), at4.GetGenderAttribute())
    print(at5.WichAttributeType(), at5.GetOrientationAttribute())
    labels = []
    for i, attr in enumerate([at1, at2, at3, at4, at5]):
        complexType = NI.ComplexLabelType(CvUtils.Human, attr)
        label = NI.Label()
        print(complexType.WichSubtype())
        label.name = str(i)
        label.position = i
        label.type = complexType
        label.reportType = CvUtils.ClassReportType.NotReport
        labels += [label]

    weights, Test = NI.DecryptAnn(r"c:\Program Files\Common Files\AxxonSoft\DetectorPack\NeuroSDK\ppeHelmet(head)General_onnx.ann")
    ttt = Test.networkParams
    for t in ttt:
        print(t.type.GetSubtypeBodySegments())
    Test.networkParams = labels
    meta = NI.GenerateMetadata(Test)
    with open("metaTest", "w") as f:
        f.write(meta)
    NI.DumpNetworkInformationToAnn(weights, Test, r"d:\ProjectGit\build\computervision-git\!bin\RelWithDebInfo\TestAttr.ann")
    print(meta)

TestNetworkInformation()