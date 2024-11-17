class OpName:
    class Binary:
        ADD = 'add'
        SUB = 'sub'
        MUL = 'mul'
        MAT_MUL = 'mat_mul'
        DIV = 'div'
        AND = 'and'
        OR = 'or'
        GT = 'gt'
        LT = 'lt'
        GTE = 'gte'
        LTE = 'lte'
        EQ = 'eq'
        NE = 'ne'

    class Unary:
        NOT = 'not'
        USUB = 'usub'
        IS_TRUE = 'is_true'
        IS_FALSE = 'is_false'

    class NDArray:
        ALL_ZEROS = 'NDArray_all_zeros'
        ALL_ONES = 'NDArray_all_ones'
        IDENTITY = 'NDArray_identity'
        SUM = 'NDArray_sum'
        ANY = 'NDArray_any'
        ALL = 'NDArray_all'
        SHAPE = 'NDArray_shape'

    class Special:
        ASSERT = 'assert'
        SLICING_ASSIGN = 'slicing_assign'
        SLICING = 'slicing'
        INPUT = 'input'
        LIST = 'list'
        NEW_LIST = 'new_list'
        LEN = 'len'
        RANGE = 'range'
        CONCAT = 'concat'
        CONSTANT = 'constant'
        READ_INT = 'read_int'
        EXPOSE_PUBLIC = 'expose_public'

    @staticmethod
    def NDArray_method_to_op_name(method: str):
        lookup = {
            'all_zeros': OpName.NDArray.ALL_ZEROS,
            'all_ones': OpName.NDArray.ALL_ONES,
            'identity': OpName.NDArray.IDENTITY,
            'sum': OpName.NDArray.SUM,
            'any': OpName.NDArray.ANY,
            'all': OpName.NDArray.ALL,
            'shape': OpName.NDArray.SHAPE,
        }
        if method in lookup:
            return lookup[method]
        raise NotImplementedError(f'Operator for `NDArray.{method}` not defined.')

    @staticmethod
    def is_supported_operator_name(op_name: str):
        return op_name in [
            OpName.Binary.ADD, OpName.Binary.SUB, OpName.Binary.MUL, OpName.Binary.DIV, OpName.Binary.AND,
            OpName.Binary.OR, OpName.Binary.MAT_MUL, OpName.Binary.EQ, OpName.Binary.NE, OpName.Binary.GT, OpName.Binary.LT,
            OpName.Binary.GTE, OpName.Binary.LTE,
            OpName.Unary.NOT, OpName.Unary.USUB, OpName.NDArray.ALL_ZEROS, OpName.Special.LIST, OpName.Special.RANGE,
            OpName.NDArray.ALL_ONES, OpName.NDArray.IDENTITY, OpName.NDArray.SUM, OpName.Special.ASSERT, OpName.Special.CONCAT,
            OpName.Special.SLICING_ASSIGN, OpName.Special.INPUT, OpName.Special.CONSTANT, OpName.Special.READ_INT,
            OpName.Special.EXPOSE_PUBLIC, OpName.NDArray.ANY, OpName.NDArray.ALL, OpName.Special.NEW_LIST,
            OpName.Special.LEN, OpName.NDArray.SHAPE,
        ]

    @staticmethod
    def is_constant_arg_operator(op_name: str) -> bool:
        return op_name in [OpName.NDArray.SUM, OpName.NDArray.ANY, OpName.NDArray.ALL]

    @staticmethod
    def is_constant_operator(op_name: str) -> bool:
        return op_name in [OpName.NDArray.ALL_ONES, OpName.NDArray.ALL_ZEROS, OpName.NDArray.IDENTITY, OpName.Special.RANGE]
