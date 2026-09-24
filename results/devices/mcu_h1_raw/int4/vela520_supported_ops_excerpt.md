### Ethos-U55 and Ethos-U65 TFLite FULLY_CONNECTED Constraints

This is a list of constraints that the FULLY_CONNECTED operator must satisfy in order to be scheduled on the NPU.

- Bias tensors must be constant
- Bias tensors precision must be Int32 or Int64
- Bias values must be stored in the channel axis
- FullyConnected weights must be on the form O,1,1,..,1,I.
- IF Matmul or (FC with dynamic weights): IFM precision must be Int8
- IF Matmul or (FC with dynamic weights): OFM depth must be less than or equal to 2^16
- IF Matmul or (FC with dynamic weights): Reduced axis must be less than or equal to 2^16
- Int64 bias must be smaller than 1099511627775
- Kernel stride must be in the range (1,3).
- Weight tensors must be 8-bit precision
- Weigth sum constraints:
  - IF 8-bit IFM:
    - The sum of absolute weights cannot exceed 8323072
  - IF 16-bit IFM:
    - The sum of absolute weights cannot exceed 8323072


| TILE | [Generic](#ethos-u55-and-ethos-u65-tosa-generic-constraints), [Specific](#ethos-u55-and-ethos-u65-tosa-tile-constraints) |
| TRANSPOSE | [Generic](#ethos-u55-and-ethos-u65-tosa-generic-constraints), [Specific](#ethos-u55-and-ethos-u65-tosa-transpose-constraints) |
| TRANSPOSE_CONV2D | [Generic](#ethos-u55-and-ethos-u65-tosa-generic-constraints), [Specific](#ethos-u55-and-ethos-u65-tosa-transpose_conv2d-constraints) |

## Ethos-U55 and Ethos-U65 TOSA Generic Constraints

This is a list of constraints that most TOSA operators must satisfy in order to be scheduled on the NPU.

- Baseline:
  - TOSA Version: v1.0
  - TOSA Profile: PRO-INT
  - TOSA Level: 8k
- PRO-INT extensions:
  - EXT-INT16: 16-bit integer operations
  - EXT-INT4: 4-bit integer weights
  - EXT-DOUBLEROUND: double rounding support for RESCALE
- Operators and configurations not listed as supported below are not supported.
- For TOSA operators, CPU fallback is not supported.

## Ethos-U55 and Ethos-U65 TOSA Specific Operator Constraints


### Ethos-U55 and Ethos-U65 TOSA CONV2D Constraints

This is a list of constraints that the CONV2D operator must satisfy in order to be scheduled on the NPU.

- DataTypes (IFM, Weights, OFM): (i8_t,i8_t,i32_t), (i8_t,i4_t,i32_t), (i16_t,i8_t,i48_t)
- EXT-INT16 is only supported if followed by RESCALE to i8_t, i16_t or i32_t
- Tensor axis: NHW any, C in range 1..65536
- Kernel size: 1 <= kernel_x*kernel_y <= 4096 and 1 <= kernel_y <= 64
- Kernel stride: 1 <= stride <= 3
- Sum of absolute weights must not exceed 127*65536
- Bias must fit in signed int40: -549755813888..549755813887

