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
        [YamlMember(Alias = "Model")]
        public Model Model { get; set; }
        [YamlMember(Alias = "Pretreatment")]
        public Pretreatment Pretreatment { get; set; }
        public Trains Trains { get; set; }
    }

    public class Model
    {
        public List<string> Sites { get; set; }
        public string ModelName { get; set; }
        public string ModelType { get; set; }
        public object CharSet { get; set; }
        public int ImageWidth { get; set; }
        public int ImageHeight { get; set; }
        public List<string> CharExclude { get; set; }
        public Dictionary<string, string> CharReplace { get; set; }
    }

    public class Pretreatment
    {
        public List<int> Resize { get; set; }
        public int Binaryzation { get; set; }
        public int Smoothing { get; set; }
        public int Blur { get; set; }
    }


    public class Trains
    {
        public string TrainsPath { get; set; }
        public string TestPath { get; set; }

    }
}
