# Possible custom errors


class DatasetException():
    pass

class ColumnMismatchException(DatasetException):
    def __init__(self, message="Make sure the columns input matches that in the given framework file."):
        self.message = message
        super().__init__(self.message)

