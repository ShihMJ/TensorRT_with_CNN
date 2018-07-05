import tensorrt as trt
import uff
from tensorrt.parsers import uffparser

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)



#uff_model = "6_24_uff.uff"

uff_model = uff.from_tensorflow_frozen_model("final.pb", ["dense_2/Softmax"])

INFERENCE_BATCH_SIZE = 256

parser = uffparser.create_uff_parser()

parser.register_input("conv2d_1_input", (1, 28, 28), 0)
parser.register_output("dense_2/Softmax")

engine = trt.utils.uff_to_trt_engine(G_LOGGER,
									uff_model,
									parser,
									INFERENCE_BATCH_SIZE,
									1<<20,
									trt.infer.DataType.FLOAT)

trt.utils.write_engine_to_file("test_engine.engine", engine.serialize())