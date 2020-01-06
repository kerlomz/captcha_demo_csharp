using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YamlDotNet.Serialization;

namespace CaptchaDemo
{
    public class Config
    {
        [YamlMember(Alias = "FieldParam")]
        public FieldParam FieldParam { get; set; }

        [YamlMember(Alias = "Model")]
        public Model Model { get; set; }
        [YamlMember(Alias = "System")]
        public ConfigSystem ConfigSystem { get; set; }
        public NeuralNet NeuralNet { get; set; }
        public Label Label { get; set; }
        public Trains Trains { get; set; }
        public DataAugmentation DataAugmentation { get; set; }
        public Pretreatment Pretreatment { get; set; }
    }

    public class ConfigSystem
    {
        public float MemoryUsage { get; set; }
        public int Version { get; set; }

    }
    public class Model
    {
        public string ModelName { get; set; }
        public string ModelField { get; set; }
        public string ModelScene { get; set; }
    }

    public class NeuralNet
    {
        public object CNNNetwork { get; set; }
        public object RecurrentNetwork { get; set; }
        public object UnitsNum { get; set; }
        public string Optimizer { get; set; }
        public object OutputLayer { get; set; }

    }


    public class FieldParam
    {
        public List<int> Resize { get; set; }
        public int ImageChannel { get; set; }
        public string ModelType { get; set; }
        public object Category { get; set; }
        public int ImageWidth { get; set; }
        public int ImageHeight { get; set; }
        public int MaxLabelNum { get; set; }
        public string OutputSplit { get; set; }
        public bool AutoPadding { get; set; }
    }

    public class Label
    {
        public object LabelFrom { get; set; }
        public object ExtractRegex { get; set; }
        public object LabelSplit { get; set; }
    }

    public class Trains
    {
        public object DatasetPath { get; set; }
        public object SourcePath { get; set; }
        public object ValidationSetNum { get; set; }
        public object SavedSteps { get; set; }
        public object ValidationSteps { get; set; }
        public object EndAcc { get; set; }
        public object EndCost { get; set; }
        public object EndEpochs { get; set; }
        public object BatchSize { get; set; }
        public object ValidationBatchSize { get; set; }
        public object LearningRate { get; set; }
    }

    public class DataAugmentation
    {
        public object Binaryzation { get; set; }
        public object MedianBlur { get; set; }
        public object GaussianBlur { get; set; }
        public object EqualizeHist { get; set; }
        public object Laplace { get; set; }
        public object WarpPerspective { get; set; }
        public object Rotate { get; set; }
        public object PepperNoise { get; set; }
        public object Brightness { get; set; }
        public object Saturation { get; set; }
        public object Hue { get; set; }
        public object Gamma { get; set; }
        public object ChannelSwap { get; set; }
        public object RandomBlank { get; set; }
        public object RandomTransition { get; set; }
    }

    public class Pretreatment
    {
        public object Binaryzation { get; set; }
        public object ReplaceTransparent { get; set; }
        public object HorizontalStitching { get; set; }
        public object ConcatFrames { get; set; }
        public object BlendFrames { get; set; }

    }
}
