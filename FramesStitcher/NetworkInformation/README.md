
 # NetworkInformation
Текущая версия: 0.2.0

####История изменений:
* 0.0.0 - Появление нового формата metadata-json
* 0.1.0 - Добавление Python Binding для NetworkInfomation
* 0.1.1 - Фикс багов
* 0.2.0 - Добавление нового функционала для Bindings. (ValidationParameters, DecryptAnn, GenerateMetadata, CreateEmptyNetParams,ConsumeWeightsData). Исправление багов в DumpNetworkInformationToAnn
* 0.2.1 - Изменение описания ошибки для устаревших моделей

 Модуль для хранения дополнительныйх параметров сети необходимых для PreProcess, PostProcess, Inference, etc

**Важно**
 [**NETWORK_INFORMATION_METADATA_VERSION**](#ItvCv::Version) в **TagsConstant.h** При изменении или добавлении в NetworkInformation нужно инкрементировать значение. Сама версия проставляется автоматически при [дампе сети](#DumpNetworkInformationToAnn).
* **major** - Изменения затрагивающие структуру json. Когда старые сети перестанут работать. Например: изменение названия тега.
* **minor** - Изменения которые расширяют функционал. И не отламывают поддержку старых сетей. Например: добавление новой топополигии
* **patch** - Изменения исправляющие ошибки, и не затрагивающие структуру json
Сети с версией **major != NETWORK_INFORMATION_METADATA_VERSION.major**   не будут работать.

**Изменение или добавление новых параметров в json**
Парсинг реализован через template. И при добавление нового параметра, нужно изменять только функцию парсящий определенные параметры - NodeType. и добавить функцию Dump которая будет делать обратную операцию
```
template<NodeType NodeType_t>
struct MetadataJson
{
  static void Parse(const Json::Value& root);
  static Json::Value Dump(Args...Args);
}
```
Все названия тегов описаны в файле **TagsConstant.h** и если изменяеться или добавляется новый тег то его следует добавлять в **TagsConstant.h**

# Оглавление

- [NetworkInformation](#структура-networkinformation)
- [PybindBuild](#pybindbuild)
- [ItvCv::Version](#itvcv::version)
- [Структура NetworkInformation](#структура-networkinformation)
- [Структура InputParams](#структура-inputparams)
- [Структура CommonParams](#структура-commonparams)
- [Структура ModelDescription](#структура-modeldescription)
- [Типы NetParams_t](#типы-netparams_t)
    - [Labels_t](#labels_t)
        - [класс ItvCv::Label](#класс-itvcv::label)
    - [ReidParams](#reidparams)
    - [SemanticSegmentationParameters](#semanticsegmentationparameters)
    - [PoseNetworkParams](#posenetworkparams)
- [класс ComplexLabelType](#класс-complexlabeltype)
- [Интерфейсные функции](#интерфейсные-функции)
    - [GetMetadataParserVersion](#getmetadataparserversion)
    - [GetNetworkInformation](#getmetadataparserversion)
    - [ConsumeWeightsData](#consumeweightsdata)
    - [DumpNetworkInformationToAnn](#dumpnetworkinformationtoann)
- [Пример использования для поз:](#пример-использования-для-поз:)

# PybindBuild

Необходима версия: Python 3.9
Необходимо выбрать следующие cmake флаги:
- BUILD_PYBIND11=ON

Так же необходимо собрать ItvCvUtilsPyBind

Если будите подключать в VS code, то нужно сгенерировать .pyi фалы, для подсказок. Если не ошибаюсь PyCharm может самостоятельно генерировать данные файлы.

Например, для себя я вынес все зависимости в отдельную папку  NetInfo и добавил \__init\__.py

Код \__init\__.py
```python
import sys
import os
dirPath = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dirPath)
import NetworkInformationPyBind
import ItvCvUtilsPyBind
from pybind11_stubgen import ModuleStubsGenerator

def GeneratePyiFiles(moduleToParse):
    namePyi = f"{dirPath}/{moduleToParse.__name__}.pyi"
    if(os.path.exists(namePyi)):
        return
    print(moduleToParse.__name__)
    module = ModuleStubsGenerator(moduleToParse)
    module.parse()
    module.write_setup_py = False
    with open(namePyi, "w") as fp:
        fp.write("#\n# AUTOMATICALLY GENERATED FILE, DO NOT EDIT!\n#\n\n")
        fp.write("\n".join(module.to_lines()))

GeneratePyiFiles(NetworkInformationPyBind)
GeneratePyiFiles(ItvCvUtilsPyBind)
```
Пример использования:
```python
from NetInfo import ItvCvUtilsPyBind as CvUtils
from NetInfo import NetworkInformationPyBind as NI
```
# ItvCv::Version

Структура описания версии модуля и metadata-json

|  Name |DataFormat   | Description  |
| ------------ | ------------ | ------------ |
| major | int |  Major версия. Изменения которые несовместимы со старыми metada-json или [NetworkInformation](#структура-networkinformation)) |
| minor | int |  Minor версия. Изменения которые не ломают обратную совместимость |
| patch | int |  Patch версия. Мелкие изменения |

# Структура NetworkInformation
Данная структура заполняется по средствам прасинга metadata-json. Она необходима для препроцессинга, постпроцессинга, и содержит параметры которые необходимы для анализа.

Описание атрибутов структуры NetworkInformation

|  Name |DataFormat   | Description  |
| ------------ | ------------ | ------------ |
| modelData  |   std::string |  описательная часть модели. Актуально только для **IR(.xml)** и **Caffe model(.caffeproto)**. Для **onnx** моделей должна быть пустой |
| inputParams  |  [ItvCv::InputParams](#структура-inputparams) |  Параметры отвечающие за препроцессин |
| commonParams |  [ItvCv::CommonParams](#структура-commonparams) |   общие параметры сети(тип архитиктуры, тип аналайзера, тип представления, формат весов)|
|networkParams | using [NetParams_t](#типы-netparams_t) = boost::variant< boost::blank, [Labels_t](#Labels_t), [SemanticSegmentationParameters](#SemanticSegmentationParameters), [PoseNetworkParams](#PoseNetworkParams), [ReidParams](#ReidParams) >| необходимые параметры для разных типов аналайзера и архитиктур сетей|
|description | boost::optional< [ModelDescription](#структура-modeldescription) >|  опциональные параметр, описание сети или дополнительное пояснение не влияющие на инференс и логику работы|

# Структура InputParams
|  Name |DataFormat   | Description  |
|------ |----- | ------|
|numChannels |  int| обязательный параметр: кол-во каналов|
|inputWidth | int| обязательный параметр:  ширина семпла|
|inputHeight | int| обязательный параметр:  высота семпла|
|supportedDynamicBatch | bool| обязательный параметр: поддержка динамического shape|
|resizePolicy | ItvCv::ResizePolicy| опциональный параметр: метод ресайза кадра, по умолчанию ResizePolicy::Unspecified|
|pixelFormat | ItvCv::PixelFormat| обязательный параметр: формат пикселей (BGR, RGB, NV12, etc)|
|normalizationValues | std::vector< ItvCv::NormalizationValue >| обязательный параметр: параметры нормализации mean, scale |

# Структура CommonParams
|  Name |DataFormat   | Description  |
|------ |----- | ------|
|architecture |  ItvCv::ArchitectureType| обязательный параметр: тип архитектуры, для некоторых типов архитектур будет отличаться PostProcessing|
|analyzerType | itvcvAnalyzerType| обязательный параметр: тип сети\обработчика. Классификация, Детекция, Сегментация, ReID,  Позы|
|modelRepresentation |  ItvCv::ModelRepresentation| обязательный параметр: тип представления сети: caffe(deprecated), onnx, IR, ascend |
| weightType|  ItvCv::DataType| обязательный параметр: тип данных весов модели. FP16, FP32, INT8|

# Структура ModelDescription
|  Name |DataFormat   | Description  |
|------ |----- | ------|
| author| std::string| опциональный параметр: индитификатор того кто последний зашифровал сеть, по умолчанию: ""|
| task| std::string| опциональный параметр: имя задачи для которой готовилась сеть. по умолчанию: ""|
| info| std::string| опциональный параметр: краткое описание сети\используемых данных\имя последнего снапшота итд. По умолчанию: "" |
|metadataVersion | [ItvCv::Version](#ItvCv::Version)| Версия NetInfo и metadata-json. **Версию указывать не нужно. Она только возврашаться. При формировании metadata-json проставляется автоматически.**|

# Типы NetParams_t

## Labels_t

Параметры **обязательные** для **классификаионных** и **детекционных** сетей.
### класс ItvCv::Label
|  Name |DataFormat   | Description  |
|------ |----- | ------|
|name | std::string | обязательный параметр: имя класса|
|position | int | обязательный параметр: позиция класса в векторе|
|type | [ItvCv::ComplexLabelType](#класс-complexlabeltype) | обязательный параметр: составной тип объекта|
|reportType | ItvCv::ClassReportType | обязательный параметр: тип класса негативный(**Report** - участвует в анализе, создает тревогу и нужно репортить), позитивный(**NotReport** - участвует в анализе, но не создает тревогу и не репортится), пропуск(**Skip** - не участвует в анализе). |

## ReidParams
Параметры для сетей ReID

|  Name |DataFormat   | Description  |
|------ |----- | ------|
|  type | [ItvCv::ComplexLabelType](#класс-complexlabeltype)|  обязательный параметр: составной тип объекта|
|  vectorSize | int|  обязательный параметр: размер вектора признаков(embedding)|
|  datasetGroup | std::string| обязательный параметр: уникальное имя дата сета на котором обучалась сеть. **ВАЖНО:** вместе с version образуют уникальный индетефикатор сети чтобы можно было отличить находятся ли признаки в одном пространстве признаков|
|  version | int|  обязательный параметр: версия сети, если сеть дообучалась, или обучалась заного с данного datasetGroup(датасета) то версию нужно увеличить.  **ВАЖНО:** вместе с datasetGroup образуют уникальный индетефикатор сети чтобы можно было отличить находятся ли признаки в одном пространстве признаков|

## SemanticSegmentationParameters
|  Name |DataFormat   | Description  |
|------ |----- | ------|
| labels| [std::vector< ItvCv::Label >](#Labels_t)|  обязательный параметр:  классы на которые обучена сеть |
| isSingleChannel | bool | обязательный параметр:  Маска многоканальная или одноканальная. В случае если **многоканальная** то ItvCv::Label::position позиция канала. В случае если **одноканальная** то значение пикселя это ItvCv::Label::position|
## PoseNetworkParams
Параметры сетей для предсказания поз

|  Name |DataFormat   | Description  |
|------ |----- | ------|
| linker| ItvCv::PoseLinker| обязательный параметр:  Параметры связывания и позиции точек|
| params| boost::variant< OpenPoseParams, AEPoseParams >| обязательный параметр:  параметры для PostProcessing. Зависят от типа архитектуры|
| minSubsetScore| float| обязательный параметр: минимальный порог предсказанной позы|
| type| [ItvCv::ComplexLabelType](#класс-complexlabeltype) | обязательный параметр: составной тип объекта|

# класс ComplexLabelType

Составной тип объекта. необходимый для понимания какой объект мы анализируем.

Атирибуты

|  Name |DataFormat   | Description  |
|------ |----- | ------|
| objectType| ObjectType| обязательный параметр: тип анализируемого объекта. (Human, Car, Fire, Noise, ..., etc)|
| subType | SubTypes_t| опциональный параметр: уточняющий подтип на текущий момент только BodySegment но планируется его расширять. Например: objectType-Human, а subType - BodySegment::Head. Означает что мы анализируем людей, и часть тела голова. Необходимо для создания более гибкой логики. Будет расширяться(я надеюсь) |

Функции:

|  Name | Args |DataFormat   | Description  |
|------ |----- |----- | ------|
| WichSubtype| void | ObjectType|  Возвращает тип Subtype|
| GetSubtype< SubTypes Type_t, typename T> | void | boost::optional< T >|  В зависимости от типа SubTypes Type_t возвращает необходимые T параметры. Если их нет или тип неправильный то optional empty|

# Интерфейсные функции

Функции предоставляемые модулем NetworkInformation

## GetMetadataParserVersion
**Args:** void
**Return:** [ItvCv::Version](#ItvCv::Version)
**Desc:** Возвращает версию модуля NetworkInformation и metadata-json

## GetNetworkInformation
**Args:**
- const char* pathToEncryptedModel - путь к модели ann

**Return:** std::shared_ptr< NetworkInformation >
**Desc:**  Возвращает данные NetworkInformation полученные из metadata-json указанной модели ann

## ConsumeWeightsData
**Args:**
- std::shared_ptr< [NetworkInformation](#структура-networkinformation) > const& netInfo- NetworkInformation полученный через GetNetworkInformation

**Return:** std::string
**Desc:**  Возвращает веса модели. Веса модели храняться отдельно от NetworkInformation

## DumpNetworkInformationToAnn
**Args:**
- const std::string& weightsData - данные весов, передается через std::string,
- const [NetworkInformation](#структура-networkinformation)& netInfo - параметры NetworkInformation которые необходимо предствавить в виде metadata-json,
- const std::string& pathOut - путь к сохроняемой подели и ее название. Пример: /path/name.ann.** Важно:** папки указанные в пути должны быть созданны.
- std::int64_t byteSize - кол-во байт для шифрования по умолчанию значение 1024. Если значение отрицательное или больше размера данных. То будет взято максимальное кол-во. От данного параметра так же зависит скорость, и размер ann.

**Return:** bool
**Desc:** Преобразует параметры NetworkInformation в metadata-json, и вместе с весами шифрует в ann файл. При неправильных параметрах вернет false
## GenerateMetadata
**Args:**
- const NetworkInformation& netInfo - параметры NetworkInformation которые необходимо предствавить в виде metadata-json,

**Return:** std::string
**Desc:** Преобразует параметры NetworkInformation в metadata-json

#Utils.h
## ValidationParameters
**Args:**
- const <
ItvCv::NetworkInformation, ItvCv::InputParams, ItvCv::CommonParams, ItvCv::NetworkInformation::Labels_t,
ItvCv::ReidParams,
ItvCv::SemanticSegmentationParameters,
ItvCv::PoseNetworkParams,
ItvCv::OpenPoseParams,
ItvCv::AEPoseParams>& data - параметры.

**Return:** std::pair<ValidationError, std::string>
**Desc:** Валидирует входные параметры, возвращая ошибку и сообщение
## Validation
**Args:**
- const <
ItvCv::NetworkInformation, ItvCv::InputParams, ItvCv::CommonParams, ItvCv::NetworkInformation::Labels_t,
ItvCv::ReidParams,
ItvCv::SemanticSegmentationParameters,
ItvCv::PoseNetworkParams,
ItvCv::OpenPoseParams,
ItvCv::AEPoseParams>& data - параметры.

**Return:** void
**Desc:** Валидирует входные параметры **генерируется exception**
## GenerateMetadata
**Args:**
- const NetworkInformation& netInfo - параметры NetworkInformation которые необходимо предствавить в виде metadata-json,

**Return:** std::string
**Desc:** Преобразует параметры NetworkInformation в metadata-json
# Пример использования для поз:
```python
............
from NetInfo import ItvCvUtilsPyBind as CvUtils
from NetInfo import NetworkInformationPyBind as NI

DICT_HUMAN_POSE_POINT_TYPE = {
'nose': CvUtils.humanPointNose,
'neck': CvUtils.humanPointNeck,
'rShoulder': CvUtils.humanPointRightShoulder,
'rElbow': CvUtils.humanPointRightElbow,
'rWrist': CvUtils.humanPointRightWrist,
'lShoulder': CvUtils.humanPointLeftShoulder,
'lElbow': CvUtils.humanPointLeftElbow,
'lWrist': CvUtils.humanPointLeftWrist,
'rHip': CvUtils.humanPointRightHip,
'rKnee': CvUtils.humanPointRightKnee,
'rAnkle': CvUtils.humanPointRightAnkle,
'lHip': CvUtils.humanPointLeftHip,
'lKnee': CvUtils.humanPointLeftKnee,
'lAnkle': CvUtils.humanPointLeftAnkle,
'rEye': CvUtils.humanPointRightEye,
'lEye': CvUtils.humanPointLeftEye,
'rEar': CvUtils.humanPointRightEar,
'lEar': CvUtils.humanPointLeftEar
}

def DefaultOpenPoseParams():
    netParameter = NI.OpenPoseParams()
    netParameter.boxSize = 368
    netParameter.stride = 8
    netParameter.minPeaksDistance = 3
    netParameter.midPointsScoreThreshold = 0.05
    netParameter.midPointsRatioThreshold = 0.8
    netParameter.upSampleRatio = 4
    return netParameter

def LinkerToNetInfo(linker:dict(), keyTypeMapper = DICT_HUMAN_POSE_POINT_TYPE):
    poseLinker = NI.PoseLinker()
    if("Paf" in linker.keys() and len(linker["Paf"]) > 0):
        paf = []
        for id, fromValue, toValue in linker["Paf"]:
            pafElem = NI.SPafElement()
            pafElem.idChannel = id
            pafElem.idPointFrom = fromValue
            pafElem.idPointTo = toValue
            paf.append(pafElem)
        poseLinker.paf = paf.copy()

    if("Heatmap" not in linker.keys()):
        return None
    count = len(linker["Heatmap"].keys())
    heatmap = [0 for i in range(count)]
    for key, value in linker["Heatmap"].items():
        heatmap[value] = int(DICT_HUMAN_POSE_POINT_TYPE[key])
    poseLinker.heatmap = heatmap.copy()
    return poseLinker

def EncryptVersion100(
    topologySetting,
    linkerSetting,
    commonSettings,
    modelRepresentation:NI.ModelRepresentation,
    waightType:NI.DataType,
    arhType:NI.ArchitectureType,
    means,
    scale,
    pathOut,
    pathToWeights,
    pathToModel=None):
    netInfo = NI.NetworkInformation()

    netInfo.commonParams.modelRepresentation = modelRepresentation
    netInfo.commonParams.architecture = arhType
    netInfo.commonParams.analyzerType = CvUtils.itvcvAnalyzerType.HumanPoseEstimator
    netInfo.commonParams.weightType = waightType

    netInfo.inputParams.inputHeight = commonSettings["ToSizeH"]
    netInfo.inputParams.inputWidth = commonSettings["ToSizeW"]
    netInfo.inputParams.supportedDynamicBatch = False
    netInfo.inputParams.numChannels = 3
    means = GetListValue(means, netInfo.inputParams.numChannels, 0)
    scale = GetListValue(scale, netInfo.inputParams.numChannels, 1)
    if(means is None or scale is None):
        print("asdasdasd")
        return
    normParam =[]
    for m,s in zip(means, scale):
        norm = NI.NormalizationValue()
        norm.scale = s
        norm.mean = m
        normParam.append(norm)
    netInfo.inputParams.normalizationValues = normParam
    netInfo.inputParams.pixelFormat = NI.PixelFormat.BGR
    netParameter = NI.PoseNetworkParams()
    if arhType == NI.ArchitectureType.Openpose18_MobileNet:
        netParameter.params = DefaultOpenPoseParams()
        netParameter.params.boxSize = netInfo.inputParams.inputHeight
    elif arhType == NI.ArchitectureType.HigherHRNet_AE:
        netParameter.params = NI.AEPoseParams()

    netParameter.linker = LinkerToNetInfo(linkerSetting)
    # netInfo.inputParams.normalizationValues = [ standart for i in range(netInfo.inputParams.numChannels)]
    netParameter.type = NI.ComplexLabelType(CvUtils.Human)
    netParameter.minSubsetScore = 0.2
    netInfo.networkParams = netParameter
    if(pathToModel is not None):
        netInfo.modelData = pathToModel
    with open(pathToWeights, "rb") as f:
        weightData = f.read()
    return NI.DumpNetworkInformationToAnn(weightData, netInfo, pathOut)
if __name__ == '__main__':
........
    try:
........
        torch.onnx.export(
            net,
            input,
            onnxPath,
            verbose=True,
            input_names=input_layer_names,
            do_constant_folding=True,
            opset_version=10,
            output_names=output_layer_names)

        EncryptVersion100(
            topologySetting,
            linker,
            commonSettings,
            NI.ModelRepresentation.onnx,
            NI.DataType.FP32,
            NI.ArchitectureType.Openpose18_MobileNet,
            0,
            1/255,
            f"{args.output_dir}/{nameEncr}.ann",
            onnxPath)
    except Exception as e:
        print(e)
        exit()

```