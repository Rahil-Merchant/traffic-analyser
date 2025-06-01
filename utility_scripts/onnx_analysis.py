import onnx

model_path = r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\models\weights\silent_contract_9082.onnx"

try:
    # Load the ONNX model
    model = onnx.load(model_path)

    input_names = [inp.name for inp in model.graph.input]

    print(f"Definitive Model Input Name(s) found in '{model_path}':")
    if not input_names:
        print("No primary inputs found. This might indicate an issue with the model or it might only have initializer inputs.")
    else:
        for name in input_names:
            print(f"- {name}")

    # For more details, you can print shape information if available
    print("\nDetailed Input Information:")
    for i, input_tensor in enumerate(model.graph.input):
        print(f"\nInput #{i}:")
        print(f"  Name: {input_tensor.name}")
        # The shape is under type.tensor_type.shape.dim
        dims = input_tensor.type.tensor_type.shape.dim
        shape_str = ", ".join([str(dim.dim_value) if dim.dim_value > 0 else dim.dim_param if dim.dim_param else "?" for dim in dims])
        print(f"  Shape (from model): [{shape_str}]")
        print(f"  Data Type (from model): {onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure the model path is correct and the ONNX file is not corrupted.")