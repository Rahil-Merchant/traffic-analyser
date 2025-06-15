import trtutils
trtutils.build_engine(default_device="gpu",
                      fp16=True,
                      workspace=7.5,
                      shapes=[("images", (64, 3, 640, 640))],
                      ignore_timing_mismatch=True,
                      timing_cache=r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\models\weights\silent_contract_9082_timing.cache",
                      onnx=r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\models\weights\silent_contract_9082.onnx",
                      output=r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\models\weights\silent_contract_9082_trtutils_bs64_2.engine",
                      verbose=True
                      )