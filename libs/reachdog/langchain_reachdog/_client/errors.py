"""The one exception a recipe raises when it stops before completing: a code and a plain message."""


class RecipeBlocked(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message
