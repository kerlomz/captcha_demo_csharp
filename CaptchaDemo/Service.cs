using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using TensorFlow;
using YamlDotNet.RepresentationModel;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace CaptchaDemo
{
    public class Service
    {
        private const string ALPHANUMERIC = "ALPHANUMERIC";
        private const string ALPHANUMERIC_LOWER = "ALPHANUMERIC_LOWER";
        private const string ALPHANUMERIC_UPPER = "ALPHANUMERIC_UPPER";
        private const string NUMERIC = "NUMERIC";
        private const string ALPHABET_LOWER = "ALPHABET_LOWER";
        private const string ALPHABET_UPPER = "ALPHABET_UPPER";

        public static List<string> charset = new List<string>();

        public static Tuple<TFSession, TFOutput, TFOutput> init = Init("./model.yaml");
        public static TFSession tf = init.Item1;
        public static TFOutput opInput = init.Item2;
        public static TFOutput opDenseDecode = init.Item3;

        public static Config config;
        public class SerializeObject
        {
            static string _filePath;

            static public void SetFilePath(string filePath)
            {
                _filePath = filePath;
            }

            static public void Serializer<T>(T obj)
            {
                StreamWriter yamlWriter = File.CreateText(_filePath);
                Serializer yamlSerializer = new Serializer();
                yamlSerializer.Serialize(yamlWriter, obj);
                yamlWriter.Close();
            }

            static public T Deserializer<T>()
            {
                if (!File.Exists(_filePath))
                {
                    throw new FileNotFoundException();
                }
                StreamReader yamlReader = File.OpenText(_filePath);
                Deserializer yamlDeserializer = new Deserializer();

                T info = yamlDeserializer.Deserialize<T>(yamlReader);
                yamlReader.Close();
                return info;
            }
        }

        private static bool ReadYaml(string configPath)
        {
            config = new Config();

            if (!System.IO.File.Exists(configPath))
            {
                Console.WriteLine("Error! Can not find ./model.yaml file!");
                return false;
            }
            else
            {
                SerializeObject.SetFilePath(configPath);
                config = SerializeObject.Deserializer<Config>();
            }
            return true;
        }

        public static Tuple<TFSession, TFOutput, TFOutput> Init(string configPath)
        {

            var status = ReadYaml(configPath);
            if (!status)
            {
                return new Tuple<TFSession, TFOutput, TFOutput>(null, new TFOutput(), new TFOutput());
            }

            if (config.FieldParam.Category.GetType() == typeof(List<object>))
            {

                charset.Add("");
                List<object> customizedCharset = (List<object>)config.FieldParam.Category;
                for (int i = 0; i < customizedCharset.Count; i++)
                {
                    charset.Add((string)customizedCharset[i]);
                }
                Console.WriteLine(customizedCharset.ToString());
            }
            else
            {
                var charsetConfig = (string)config.FieldParam.Category;
                switch (charsetConfig) {
                    case ALPHANUMERIC:
                        charset = Constants.Alphanumeric;
                        break;
                    case ALPHANUMERIC_LOWER:
                        charset = Constants.AlphanumericLower;
                        break;
                    case ALPHANUMERIC_UPPER:
                        charset = Constants.AlphanumericUpper;
                        break;
                    case ALPHABET_UPPER:
                        charset = Constants.AlphabetUpper;
                        break;
                    case ALPHABET_LOWER:
                        charset = Constants.AlphabetLower;
                        break;
                    case NUMERIC:
                        charset = Constants.Numeric;
                        break;
                    default:
                        charset = Constants.AlphanumericLower;
                        break;
                }
                
            }

            var graph = new TFGraph();
            var file = File.ReadAllBytes(string.Format("./{0}.pb", config.Model.ModelName));
            graph.Import(file);
            var opInput = graph["input"][0];
            var opDenseDecode = graph["dense_decoded"][0];
            var tf = new TFSession(graph);
            return new Tuple<TFSession, TFOutput, TFOutput>(tf, opInput, opDenseDecode);

        }
        public static string Predict(byte[] bytes)
        {
            var img_tensor = FormatJPEG(bytes);
            if (img_tensor != null)
            {
                var runner = tf.GetRunner();
                runner.AddInput(opInput, img_tensor);
                runner.Fetch(opDenseDecode);
                var output = runner.Run();
                TFTensor result = output[0];
                StringBuilder stringBuffer = new StringBuilder();

                foreach (int s in (System.Int64[,])result.GetValue())
                {
                    if (s > charset.Count - 1)
                    {
                        Console.WriteLine("Current character set do not match the model.");
                        break;
                    }
                    stringBuffer.Append(charset[s]);
                }
                return stringBuffer.ToString();
            }
            Console.WriteLine("Imgage error!");
            return null;
        }

        static void Main(string[] args)
        {
            string path = "test.jpg";
            byte[] ImgBytes = GetPictureData(path);
            string result = Predict(ImgBytes);
            Console.WriteLine(result);
        }

        public static byte[] GetPictureData(string ImagePath)
        {
            FileStream fs = new FileStream(ImagePath, FileMode.Open);
            byte[] byData = new byte[fs.Length];
            fs.Read(byData, 0, byData.Length);
            fs.Close();
            return byData;
        }
        public static TFGraph ImConvertGraph(TFTensor image, out TFOutput input, out TFOutput output)
        {
            var graph = new TFGraph();
            input = graph.Placeholder(TFDataType.String);
            List<int> resize = config.FieldParam.Resize != null ? config.FieldParam.Resize : new List<int> { config.FieldParam.ImageWidth, config.FieldParam.ImageHeight };
            int W = resize[0];
            int H = resize[1];
            TFOutput src = graph.Cast(graph.DecodePng(contents: input, channels: 1), DstT: TFDataType.Float);
            TFOutput origin = graph.Reshape(src, graph.Const(new int[] { H, W }));
            TFOutput swapaxes = graph.Transpose(origin, graph.Const(new int[] { 1, 0 }));
            output = graph.Div(
                graph.Reshape(swapaxes, graph.Const(new int[] { W, H, 1 })),
                y: graph.Const((float)255)
                );
            output = graph.ExpandDims(output, dim: graph.Const(0));
            return graph;
        }
        public static TFTensor FormatJPEG(byte[] d)
        {
            var g = TFTensor.CreateString(d);
            TFOutput input, output;
            using (var graph = ImConvertGraph(g, out input, out output))
            {
                using (var session = new TFSession(graph))
                {
                    var runner = session.GetRunner();
                    runner.AddInput(input, g);
                    var r = runner.Run(output);
                    return r;
                }
            }
        }
    }
}
