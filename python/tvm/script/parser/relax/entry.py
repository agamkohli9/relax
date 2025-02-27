# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring, invalid-name
import inspect
from typing import Callable as _Callable
from typing import List, Optional
from typing import TypeVar as _TypeVar
from typing import Union

from tvm.ir import FuncType, TypeConstraint, TypeVar
from tvm.relax import DynTensorType, Expr, Function
from tvm.relax import Tuple as RxTuple
from tvm.relax import Type, Var
from tvm.runtime import ObjectGeneric
from tvm.tir import PrimExpr

from ...ir_builder.relax import ShapedType, tensor, create_shaped_tuple
from .._core import parse, utils

FType = _TypeVar("FType", bound=_Callable)


def function(f: FType) -> Union[Function, FType]:
    if not inspect.isfunction(f):
        raise TypeError(f"Expect a function, but got: {f}")
    if utils.is_defined_in_class(inspect.stack(), f):
        return f
    return parse(f, utils.inspect_function_capture(f))


setattr(function, "dispatch_token", "relax")


############################### R.Tensor ###############################


class TensorProxy(ObjectGeneric):
    def __call__(
        self,
        shape: Optional[List[Union[PrimExpr, str]]] = None,
        dtype: str = None,
        ndim: int = -1,
    ) -> ShapedType:
        # scalar tensor case
        if shape is not None and len(shape) == 0:
            shape = []
        if isinstance(shape, str) and dtype is None:
            dtype = shape
            shape = None
        return tensor(shape, dtype, ndim)

    def __getitem__(self, keys) -> Var:
        return self(*keys)  # pylint: disable=no-member # type: ignore

    def asobject(self):
        """Convert to object when direct call `R.Tensor`
        e.g. `x = R.invoke_closure(clo, (y,), type_args=R.Tensor)`
        """
        return DynTensorType()


Tensor = TensorProxy()  # pylint: disable=invalid-name

############################## R.Callable ##############################


class CallableProxy:
    """Function type.

    A function type consists of a list of type parameters to enable
    the definition of generic functions,
    a set of type constraints which we omit for the time being,
    a sequence of argument types, and a return type.

    We can informally write them as:
    `forall (type_params), (arg_types) -> ret_type where type_constraints`

    Parameters
    ----------
    arg_types : List[Union[Type, ShapedType]]
        The argument types

    ret_type : Type
        The return type.

    type_params : Optional[List[TypeVar]]
        The type parameters

    type_constraints : Optional[List[TypeConstraint]]
        The type constraints.
    """

    def __call__(
        self,
        arg_types: List[Union[Type, ShapedType]],
        ret_type: Type,
        type_params: Optional[List[TypeVar]] = None,
        type_constraints: Optional[List[TypeConstraint]] = None,
    ) -> FuncType:
        if isinstance(arg_types, ShapedType):
            arg_types = [arg_types]
        arg_types = [_convert_type(ty) for ty in arg_types]
        ret_type = _convert_type(ret_type)
        return FuncType(arg_types, ret_type, type_params, type_constraints)

    def __getitem__(self, keys) -> Var:
        return self(*keys)  # pylint: disable=no-member # type: ignore


Callable = CallableProxy()

############################### R.Tuple ################################


class TupleProxy:
    """The type of tuple values.

    Parameters
    ----------
    fields : List[Type]
        The fields in the tuple
    """

    def __call__(
        self,
        *fields: List[Union[Expr, Type, ShapedType]],
    ) -> Union[Expr, ShapedType]:
        if len(fields) == 1 and isinstance(fields[0], (tuple, list)):
            fields = fields[0]

        if all([isinstance(f, Expr) for f in fields]):
            return RxTuple(fields)
        elif all([isinstance(f, (ShapedType, Type, TensorProxy)) for f in fields]):
            types = [_convert_type(ty) for ty in fields]
            shapes = [ty.shape if isinstance(ty, ShapedType) else None for ty in fields]
            return create_shaped_tuple(types, shapes)
        else:
            raise TypeError(f"Invalid tuple type: {fields}")

    def __getitem__(self, keys) -> Var:
        return self(*keys)  # pylint: disable=no-member # type: ignore


Tuple = TupleProxy()

############################ R.match_shape #############################
class MatchShapePair:
    value: Expr
    pattern: List[PrimExpr]

    def __init__(self, value: Expr, pattern: List[PrimExpr]) -> None:
        self.value = value
        self.pattern = pattern


def match_shape(value: Expr, pattern: List[PrimExpr]):
    if value is None:
        raise ValueError("value of match_shape cannot be None")
    if pattern is None:
        raise ValueError("pattern of match_shape cannot be None")
    return MatchShapePair(value, pattern)


################################ utils #################################


def _convert_type(ty: Union[Type, ShapedType, TensorProxy]) -> Type:
    if isinstance(ty, TensorProxy):
        return ty().type
    if isinstance(ty, ShapedType):
        return ty.type
    elif isinstance(ty, Type):
        return ty
    else:
        raise TypeError(f"Expect a Type or ShapedType, but got: {ty}")
