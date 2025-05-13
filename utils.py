import inspect

def unimplemented():
    """
    呼び出された関数の名前とともに NotImplementedError を送出します。
    """
    frame = inspect.currentframe().f_back
    if frame:
        function_name = frame.f_code.co_name
        line_number = frame.f_lineno
        file_name = frame.f_code.co_filename
        raise NotImplementedError(f"{function_name}() is not implemented. (File: {file_name}:{line_number})")
    else:
        raise NotImplementedError("No function name available for unimplemented.")

