# Dynamic Array Lowering to zkRAM in Zinnia

## 1. Scope and Intent

This document explains how dynamic arrays are represented and compiled in Zinnia, with emphasis on how major dynamic-array operators are lowered into memory-oriented instructions suitable for zero-knowledge backends.

The target audience is researchers and engineers who need an implementation-level understanding of:

- bounded dynamic array representation
- operator lowering strategy
- memory semantics and trace generation
- correctness constraints enforced during compilation


## 2. Dynamic Array Representation Model

### 2.1 Value-level model

Dynamic arrays are represented by DynamicNDArrayValue, which combines:

- bounded storage envelope
  - max_length
  - max_rank
  - dtype
- logical view metadata
  - logical_shape
  - logical_offset
  - logical_strides
- runtime metadata (symbolic or constant IntegerValue)
  - runtime_logical_length
  - runtime_rank
  - runtime_shape_entries
  - runtime_stride_entries
  - runtime_offset

The key design is: data storage is always bounded and flat, while logical and runtime metadata describe the active view.

### 2.2 Type-level model

The static type descriptor is DynamicNDArrayDTDescriptor(dtype, max_length, max_rank).

This creates two layers:

- compile-time safety envelope from type descriptor
- runtime interpretation from metadata witnesses


## 3. zkRAM Instruction Substrate

Dynamic array lowering uses three memory IR primitives:

- AllocateMemoryIR(segment_id, size, init_value)
- WriteMemoryIR(segment_id, address, value)
- ReadMemoryIR(segment_id, address)

These are emitted through IRBuilder methods:

- ir_allocate_memory
- ir_write_memory
- ir_read_memory

In execution, each segment is a finite map initialized with init_value and checked for address bounds.


## 4. Generic Lowering Pattern

Many dynamic operators follow the same pattern:

1. Materialize source values into a fresh zkRAM segment.
2. Compute runtime-safe addresses under logical metadata.
3. Perform read or read-modify-write loops with bounded iteration.
4. Build an output bounded vector.
5. Return a new DynamicNDArrayValue with updated metadata.

This gives oblivious, fixed-shape circuit structure while preserving dynamic behavior through conditions and metadata constraints.


## 5. Major Operator Lowering

## 5.1 Dynamic get item and slice

Component: DynamicNDArray_GetItemOp.

Supported forms:

- dynamic scalar index
- dynamic 1D slice tuple start:stop:step

Notable lowering details:

- source vector is first written into memory segment
- negative index normalization is implemented with select
- range safety is enforced with op_assert
- scalar result reads one memory cell and casts back to dtype
- slice result uses bounded loop, guarded in_range predicate, and write_ptr compaction

Result of slicing is a bounded DynamicNDArrayValue where runtime_logical_length is write_ptr.


## 5.2 Boolean mask filtering

Component: DynamicNDArray_FilterOp.

Lowering behavior:

- iterates over bounded max_len
- computes mask predicate per slot
- compacts selected elements into output using write_ptr and select-based writes

Important semantic update:

- for dynamic masks, effective mask length is runtime_logical_length, not full bounded payload length
- mask index i is active only when i < mask_runtime_len

This allows mixed static and dynamic masking patterns while keeping bounded loops.


## 5.3 Concatenate

Component: DynamicNDArray_ConcatenateOp.

Lowering behavior:

- validates dtype consistency
- derives valid axes from logical shapes
- computes out_shape for each candidate axis
- for each output linear index:
  - decode output coordinates
  - identify source array by axis prefix ranges
  - map output coordinate to source coordinate
  - encode source linear address using source logical strides and offset
  - fetch value and write to output segment

Dynamic axis support:

- if axis is runtime dynamic, per-axis candidate values are computed and selected with op_select.


## 5.4 Stack

Component: DynamicNDArray_StackOp.

Lowering behavior:

- validates dtype consistency
- computes broadcast-compatible base shape across inputs (left-padded singleton alignment)
- generates candidate out_shape for each axis in 0..rank
- for each output element:
  - decode output coords
  - choose source index from stacked axis
  - map base coords into each source using singleton-broadcast rule
  - encode source linear address and fetch
  - write selected value

Dynamic axis support:

- axis can be symbolic and is resolved by select across candidate axis layouts.


## 5.5 Dynamic broadcast binary kernel

Component: dynamic_broadcast_binary.

This is the central kernel used by arithmetic and logical binary operators when either side is dynamic.

Pipeline:

1. Normalize and assert runtime metadata constraints on both inputs.
2. Align ranks by left-padding shape and stride with singleton-friendly defaults.
3. Assert per-dimension broadcast compatibility.
4. Compute symbolic output shape and output logical length.
5. Assert output length within bounded max_length product envelope.
6. Materialize both inputs into memory segments.
7. For each bounded output slot:
   - compute active predicate i < out_len
   - decode output coordinates
   - derive lhs and rhs coordinates with singleton collapse
   - encode lhs and rhs addresses with runtime offset and stride
   - assert active addresses are in bounds
   - guarded memory reads
   - apply operator lambda
   - guarded write into output segment
8. Compute output runtime strides and return DynamicNDArrayValue with runtime metadata.

This gives true runtime broadcasting semantics within fixed circuit bounds.


## 5.6 Constructors and initializers

### Zeros and ones

Components: DynamicNDArray_ZerosOp, DynamicNDArray_OnesOp.

Lowering strategy:

- infer bounded envelope from requested shape
- directly construct bounded constant vector in IR values
- return DynamicNDArrayValue without explicit memory pass

### Eye

Component: DynamicNDArray_EyeOp.

Lowering strategy:

- allocate memory segment of bounded length
- perform bounded diagonal write logic:
  - compute diagonal addresses from n and m
  - guard writes with in_bounds and is_diag
  - route inactive writes to placeholder address/value
- read back full bounded segment into output vector

This pattern provides dynamic n and m support with bounded, oblivious loops.


## 5.7 Transpose and moveaxis

Component: DynamicNDArray_TransposeOp (moveaxis delegates to transpose).

Two modes:

- static axes: metadata permutation only (shape and stride remap)
- dynamic axes: evaluate all axis permutations and select per element

In dynamic-axes mode, no extra memory segment is required; selection occurs over precomputed candidate value vectors.


## 5.8 Aggregations

Base component: DynamicAbstractAggregator.

Lowering strategy:

- reduction itself can be direct over bounded vectors
- additionally emits oblivious memory trace events via a dedicated trace segment
  - allocate trace memory
  - for each source element, compute target reduced coordinate
  - issue read then write at target address (trace shaping)

Axis handling:

- static axis: direct reduction loop
- dynamic axis: compute candidate reductions for each axis and select by runtime axis

Concrete ops include sum, prod, min, max, argmin, argmax, any, all.


## 6. Dispatch and Integration Layer

High-level np-like operators route to dynamic operators whenever any operand is DynamicNDArrayValue.

Examples:

- np.concatenate and np.stack perform dynamic offload in mixed static/dynamic inputs
- arithmetic and logical binary abstract operators call dynamic_broadcast_binary when needed

Static arrays may be converted to dynamic arrays on-demand via to_dynamic_ndarray for mixed execution paths.


## 7. Memory Lowering Passes and Backend View

DynamicNDArrayMemoryLoweringIRPass rewrites symbolic dynamic get/set memory IR to concrete ReadMemoryIR and WriteMemoryIR primitives.

Backend integration example (Halo2 builder):

- memory segments are tracked
- read and write events are converted into memory trace tuples
- optional explicit MemoryTraceEmitIR events are supported
- traces include segment, address, timestamp, value, and read/write flag

This decouples operator semantics from backend trace encoding.


## 8. Safety and Correctness Invariants

Key invariants enforced during lowering:

- runtime rank and offset bounds are asserted
- runtime shape and stride entries are bounded and positive
- active memory addresses are bounds-checked
- dynamic axis normalization is checked with assertions
- dynamic mask runtime length is constrained to payload capacity

These assertions are first-class IR constraints and contribute to proof soundness.


## 9. Complexity and Circuit Cost Intuition

Because loops are bounded by max_length and max_rank:

- compile-time complexity scales with envelopes, not realized runtime length
- runtime dynamism is encoded as predicates and select operations
- memory-intensive operators have O(max_length) or O(max_length * max_rank) style cost

Broadly:

- get_item scalar: O(max_length) setup + O(1) read
- get_item slice/filter: O(max_length^2) style compaction due to select-based write_ptr placement
- concat/stack/broadcast: O(output_bound * rank) coordinate math plus memory IO
- reductions: O(max_length) to O(max_length * rank), plus trace shaping work


## 10. Summary

Zinnia dynamic arrays are implemented as bounded flat storage plus runtime view metadata, with operators lowered to fixed-shape zkRAM programs. The design systematically trades variable-size control flow for predicate-gated memory access and select-based data movement, which is compatible with zero-knowledge proving constraints while preserving dynamic semantics within declared bounds.
