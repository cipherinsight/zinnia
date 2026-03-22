# Type System and Build-Time Phi Lattice in Zinnia

## 1. Scope

This document explains the current Zinnia type system and control-flow type merging design, including:

- descriptor-based static typing
- implicit cast and alignment rules
- operator dispatch by namespace and receiver type
- build-time Phi-style variable merging at branch joins
- lattice rules for merge compatibility

The focus is implementation-level behavior used by the compiler today.


## 2. Type System Architecture

## 2.1 Descriptor core

All types are represented by DTDescriptor subclasses. Equality is structural at descriptor level.

Important descriptors:

- scalar descriptors
  - IntegerDTDescriptor
  - FloatDTDescriptor
  - BooleanDTDescriptor
  - NumberDTDescriptor as numeric super-family marker
- container descriptors
  - NDArrayDTDescriptor(shape, dtype)
  - DynamicNDArrayDTDescriptor(dtype, max_length, max_rank)
  - ListDTDescriptor(elements_type)
  - TupleDTDescriptor(elements_type)
- others
  - NoneDTDescriptor
  - ClassDTDescriptor
  - PoseidonHashedDTDescriptor


## 2.2 Static vs dynamic ndarray typing

Static ndarray type carries exact shape and dtype.

Dynamic ndarray type carries bounded envelope only:

- max_length
- max_rank
- dtype

Runtime shape information lives in value metadata, not in the descriptor.


## 2.3 Type registration and annotation parsing

DTDescriptorFactory maps annotation names to descriptor constructors. This is where annotation-time validation happens.

Examples of validation:

- NDArray requires dtype and integer dimensions
- DynamicNDArray requires dtype, positive max_length, positive max_rank


## 2.4 Operator dispatch

Operators are resolved from namespace plus receiver typename:

- global functions
- np-like namespace
- NDArray namespace
- DynamicNDArray namespace
- list and tuple namespaces

Dispatch relies on descriptor typenames and is implemented in operator factory tables.


## 3. Value Layer and Type Locking

Values are runtime holders over ValueStore plus a type_locked flag.

IRContext.get reconstructs typed Value objects from stores and can mark them type-locked based on scope rules.

Type locking is used to prevent unsafe cross-path reassignments when branch outcomes are not statically known.


## 4. Implicit Conversion Rules

## 4.1 Implicit cast

ImplicitTypeCastOp verifies if source type can be cast to destination type.

Highlights:

- allowed scalar cast edges include Integer to Float, Boolean to Integer, Boolean to Float
- disallowed edge includes Float to Integer in implicit cast verification
- recursive support for tuple and list casting
- ndarray-related cast supports shape-preserving dtype change


## 4.2 Implicit align

ImplicitTypeAlignOp aligns two operands to a common type for binary operations and select.

Highlights:

- scalar alignment promotes to Float when needed
- tuple and list alignment is element-wise
- ndarray alignment requires compatible shapes and aligns dtypes
- container versus ndarray alignment supports list or tuple conversion to ndarray in specific shape-compatible cases


## 5. Build-Time Phi Merging at Control-Flow Joins

## 5.1 Design goal

Phi is implemented as a build-time compiler component, not as a dedicated IR instruction.

At each if-join:

1. collect branch-local symbol updates
2. compute per-symbol type join
3. lift both branch values to join type
4. emit select on condition when needed
5. update outer scope symbol table


## 5.2 Branch-local writes and merge source maps

ConditionalScope stores writes locally. It does not immediately mutate parent scope.

At join, IRGenerator reads local var tables from true and false branch scopes and merges with parent fallback values.


## 5.3 One-branch-local-definition rule

If a variable is absent in parent and appears in only one branch:

- allowed only if that defining branch condition is statically true
- otherwise rejected

This prevents undefined-on-some-path variables from escaping dynamic branches.


## 5.4 Merge implementation flow

For each merged variable name:

- choose branch values and parent fallback if needed
- compute join type via lattice function
- reject if no join
- lift each side to join type
- choose merged value:
  - true value if condition statically true
  - false value if condition statically false
  - select(cond, true_value, false_value) otherwise


## 6. Lattice Definition in Current Compiler

## 6.1 Family partition

The join function first classifies types into families:

- scalar
- ndarray
- list
- tuple
- other

Cross-family join is rejected by design.

This means scalar and ndarray are mutually incompatible for lattice join, and similarly list and tuple are not interchangeable.


## 6.2 Scalar join

Join order:

- Boolean and Integer join to Integer
- Integer and Float join to Float
- Boolean and Float join to Float
- equal types join to themselves

No join implies incompatibility.


## 6.3 Ndarray family join

Inputs can be static ndarray descriptors or dynamic ndarray descriptors.

Rules:

- static plus static
  - shapes must match exactly
  - dtype joins via scalar join
  - result is static NDArrayDTDescriptor
- any merge involving dynamic ndarray
  - dtype joins via scalar join
  - bounds widen by max over both operands
    - max_length = max(lhs_len, rhs_len)
    - max_rank = max(lhs_rank, rhs_rank)
  - result is DynamicNDArrayDTDescriptor

Static ndarray contributes len as product of shape and rank as shape length.


## 6.4 List and tuple join

- same container family required
- same arity or length required
- elementwise join required for each position

Any elementwise failure rejects the whole merge.


## 7. Lifting Values to Joined Type

After a successful type join, each branch value is lifted:

- scalar destinations
  - bool cast, int cast, float cast as needed
- list and tuple destinations
  - recursive elementwise lifting
- static ndarray destinations
  - require static ndarray value and exact shape
  - dtype cast through ndarray astype if needed
- dynamic ndarray destinations
  - static ndarray values are promoted to dynamic via to_dynamic_ndarray
  - dynamic dtype is cast via DynamicNDArray_AsTypeOp when needed
  - bounded payload is widened to destination max_length
  - runtime shape and stride metadata are padded to destination max_rank


## 8. Assignment-Time Safety Rule Under Branch Uncertainty

When assigning to existing variables in uncertain control flow:

- if variable is type-locked and no lattice join exists between old and new type, assignment is rejected
- if type-locked and join exists, assignment can proceed via merge machinery

This preserves the paper-level rule:

- rebinding is allowed only when type consistency across paths is maintained by the lattice


## 9. Accepted and Rejected Merge Scenarios

## 9.1 Accepted examples

- Integer versus Float across branches, merged to Float
- static ndarray shape (2, 3) Integer versus static ndarray shape (2, 3) Float, merged static ndarray Float
- static ndarray versus dynamic ndarray with compatible dtype join, merged dynamic ndarray with widened bounds
- list or tuple branches with same length and elementwise-joinable members

## 9.2 Rejected examples

- scalar versus ndarray across branches
- list versus tuple across branches
- static ndarray shape mismatch across branches
- one-branch local definition when branch condition is dynamic and variable absent in parent


## 10. Interaction with Operator Semantics

Because select operation itself requires alignable operand types, the lattice-join-before-select architecture is crucial:

- merge computes a legal common type first
- both operands are lifted to that type
- select can be emitted safely

This avoids ad hoc type coercion at select sites and centralizes branch-merge typing policy.


## 11. Practical Implications for Academic Framing

The current implementation can be described as:

- descriptor-based static typing with bounded dynamic-array envelopes
- explicit cast and alignment algebra
- build-time Phi over branch environments
- finite join-semilattice over same-family types
- strict incompatibility for cross-family merges

This gives deterministic, proof-friendly typing while still supporting dynamic array shape evolution and branch-sensitive rebinding when consistency constraints are satisfied.


## 12. Current Limitations and Research Directions

Current limitations:

- no cross-family unions, by design
- one-branch new variable is rejected unless statically guaranteed
- no path-sensitive liveness relaxation for otherwise safe rebinding

Possible future extensions:

- optional union-like descriptor family for advanced Python compatibility
- liveness or dominance-based relaxation for dead-type paths
- richer lattice relations for container interoperability

These can be layered without changing the core build-time Phi strategy.
