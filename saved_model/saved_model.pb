��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
Adam/v/dense_59/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_59/bias/*
dtype0*
shape:*%
shared_nameAdam/v/dense_59/bias
y
(Adam/v/dense_59/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_59/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_59/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_59/bias/*
dtype0*
shape:*%
shared_nameAdam/m/dense_59/bias
y
(Adam/m/dense_59/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_59/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_59/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_59/kernel/*
dtype0*
shape
: *'
shared_nameAdam/v/dense_59/kernel
�
*Adam/v/dense_59/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_59/kernel*
_output_shapes

: *
dtype0
�
Adam/m/dense_59/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_59/kernel/*
dtype0*
shape
: *'
shared_nameAdam/m/dense_59/kernel
�
*Adam/m/dense_59/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_59/kernel*
_output_shapes

: *
dtype0
�
Adam/v/dense_58/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_58/bias/*
dtype0*
shape: *%
shared_nameAdam/v/dense_58/bias
y
(Adam/v/dense_58/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_58/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_58/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_58/bias/*
dtype0*
shape: *%
shared_nameAdam/m/dense_58/bias
y
(Adam/m/dense_58/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_58/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_58/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_58/kernel/*
dtype0*
shape
:  *'
shared_nameAdam/v/dense_58/kernel
�
*Adam/v/dense_58/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_58/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_58/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_58/kernel/*
dtype0*
shape
:  *'
shared_nameAdam/m/dense_58/kernel
�
*Adam/m/dense_58/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_58/kernel*
_output_shapes

:  *
dtype0
�
Adam/v/dense_57/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_57/bias/*
dtype0*
shape: *%
shared_nameAdam/v/dense_57/bias
y
(Adam/v/dense_57/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_57/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_57/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_57/bias/*
dtype0*
shape: *%
shared_nameAdam/m/dense_57/bias
y
(Adam/m/dense_57/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_57/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_57/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_57/kernel/*
dtype0*
shape
: *'
shared_nameAdam/v/dense_57/kernel
�
*Adam/v/dense_57/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_57/kernel*
_output_shapes

: *
dtype0
�
Adam/m/dense_57/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_57/kernel/*
dtype0*
shape
: *'
shared_nameAdam/m/dense_57/kernel
�
*Adam/m/dense_57/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_57/kernel*
_output_shapes

: *
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_59/biasVarHandleOp*
_output_shapes
: *

debug_namedense_59/bias/*
dtype0*
shape:*
shared_namedense_59/bias
k
!dense_59/bias/Read/ReadVariableOpReadVariableOpdense_59/bias*
_output_shapes
:*
dtype0
�
dense_59/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_59/kernel/*
dtype0*
shape
: * 
shared_namedense_59/kernel
s
#dense_59/kernel/Read/ReadVariableOpReadVariableOpdense_59/kernel*
_output_shapes

: *
dtype0
�
dense_58/biasVarHandleOp*
_output_shapes
: *

debug_namedense_58/bias/*
dtype0*
shape: *
shared_namedense_58/bias
k
!dense_58/bias/Read/ReadVariableOpReadVariableOpdense_58/bias*
_output_shapes
: *
dtype0
�
dense_58/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_58/kernel/*
dtype0*
shape
:  * 
shared_namedense_58/kernel
s
#dense_58/kernel/Read/ReadVariableOpReadVariableOpdense_58/kernel*
_output_shapes

:  *
dtype0
�
dense_57/biasVarHandleOp*
_output_shapes
: *

debug_namedense_57/bias/*
dtype0*
shape: *
shared_namedense_57/bias
k
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
_output_shapes
: *
dtype0
�
dense_57/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_57/kernel/*
dtype0*
shape
: * 
shared_namedense_57/kernel
s
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel*
_output_shapes

: *
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_57/kerneldense_57/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_586250

NoOpNoOp
�0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�/
value�/B�/ B�/
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1
	dropout1


dense2
dropout2
output_layer
	optimizer

metric

signatures*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

kernel
bias*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_random_generator* 
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

kernel
bias*
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8_random_generator* 
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

kernel
bias*
�
?
_variables
@_iterations
A_learning_rate
B_index_dict
C
_momentums
D_velocities
E_update_step_xla*
* 

Fserving_default* 
OI
VARIABLE_VALUEdense_57/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_57/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_58/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_58/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_59/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_59/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
	1

2
3
4*

G0*
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

Mtrace_0* 

Ntrace_0* 
* 
* 
* 
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 

Ttrace_0
Utrace_1* 

Vtrace_0
Wtrace_1* 
* 

0
1*

0
1*
* 
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
* 
* 
* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

dtrace_0
etrace_1* 

ftrace_0
gtrace_1* 
* 

0
1*

0
1*
* 
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

mtrace_0* 

ntrace_0* 
b
@0
o1
p2
q3
r4
s5
t6
u7
v8
w9
x10
y11
z12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
o0
q1
s2
u3
w4
y5*
.
p0
r1
t2
v3
x4
z5*
* 
* 
8
{	variables
|	keras_api
	}total
	~count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
a[
VARIABLE_VALUEAdam/m/dense_57/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_57/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_57/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_57/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_58/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_58/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_58/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_58/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_59/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_59/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_59/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_59/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*

}0
~1*

{	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_57/kerneldense_57/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/bias	iterationlearning_rateAdam/m/dense_57/kernelAdam/v/dense_57/kernelAdam/m/dense_57/biasAdam/v/dense_57/biasAdam/m/dense_58/kernelAdam/v/dense_58/kernelAdam/m/dense_58/biasAdam/v/dense_58/biasAdam/m/dense_59/kernelAdam/v/dense_59/kernelAdam/m/dense_59/biasAdam/v/dense_59/biastotalcountConst*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_586518
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_57/kerneldense_57/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/bias	iterationlearning_rateAdam/m/dense_57/kernelAdam/v/dense_57/kernelAdam/m/dense_57/biasAdam/v/dense_57/biasAdam/m/dense_58/kernelAdam/v/dense_58/kernelAdam/m/dense_58/biasAdam/v/dense_58/biasAdam/m/dense_59/kernelAdam/v/dense_59/kernelAdam/m/dense_59/biasAdam/v/dense_59/biastotalcount*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_586593ָ
�

e
F__inference_dropout_39_layer_call_and_return_conditional_losses_586339

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
d
F__inference_dropout_39_layer_call_and_return_conditional_losses_586344

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
D__inference_dense_58_layer_call_and_return_conditional_losses_586082

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
D__inference_dense_59_layer_call_and_return_conditional_losses_586364

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
$__inference_signature_wrapper_586250
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_586040o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name586246:&"
 
_user_specified_name586244:&"
 
_user_specified_name586242:&"
 
_user_specified_name586240:&"
 
_user_specified_name586238:&"
 
_user_specified_name586236:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
D__inference_dense_59_layer_call_and_return_conditional_losses_586111

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
L__inference_my_classifier_19_layer_call_and_return_conditional_losses_586118
input_1!
dense_57_586054: 
dense_57_586056: !
dense_58_586083:  
dense_58_586085: !
dense_59_586112: 
dense_59_586114:
identity�� dense_57/StatefulPartitionedCall� dense_58/StatefulPartitionedCall� dense_59/StatefulPartitionedCall�"dropout_38/StatefulPartitionedCall�"dropout_39/StatefulPartitionedCall�
 dense_57/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_57_586054dense_57_586056*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_586053�
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_586070�
 dense_58/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0dense_58_586083dense_58_586085*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_586082�
"dropout_39/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0#^dropout_38/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_586099�
 dense_59/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0dense_59_586112dense_59_586114*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_586111x
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall#^dropout_39/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall2H
"dropout_39/StatefulPartitionedCall"dropout_39/StatefulPartitionedCall:&"
 
_user_specified_name586114:&"
 
_user_specified_name586112:&"
 
_user_specified_name586085:&"
 
_user_specified_name586083:&"
 
_user_specified_name586056:&"
 
_user_specified_name586054:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

e
F__inference_dropout_39_layer_call_and_return_conditional_losses_586099

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

e
F__inference_dropout_38_layer_call_and_return_conditional_losses_586070

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
L__inference_my_classifier_19_layer_call_and_return_conditional_losses_586149
input_1!
dense_57_586121: 
dense_57_586123: !
dense_58_586132:  
dense_58_586134: !
dense_59_586143: 
dense_59_586145:
identity�� dense_57/StatefulPartitionedCall� dense_58/StatefulPartitionedCall� dense_59/StatefulPartitionedCall�
 dense_57/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_57_586121dense_57_586123*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_586053�
dropout_38/PartitionedCallPartitionedCall)dense_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_586130�
 dense_58/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0dense_58_586132dense_58_586134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_586082�
dropout_39/PartitionedCallPartitionedCall)dense_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_586141�
 dense_59/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0dense_59_586143dense_59_586145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_586111x
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:&"
 
_user_specified_name586145:&"
 
_user_specified_name586143:&"
 
_user_specified_name586134:&"
 
_user_specified_name586132:&"
 
_user_specified_name586123:&"
 
_user_specified_name586121:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
d
F__inference_dropout_38_layer_call_and_return_conditional_losses_586130

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
D__inference_dense_58_layer_call_and_return_conditional_losses_586317

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
1__inference_my_classifier_19_layer_call_fn_586166
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_my_classifier_19_layer_call_and_return_conditional_losses_586118o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name586162:&"
 
_user_specified_name586160:&"
 
_user_specified_name586158:&"
 
_user_specified_name586156:&"
 
_user_specified_name586154:&"
 
_user_specified_name586152:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

e
F__inference_dropout_38_layer_call_and_return_conditional_losses_586292

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
)__inference_dense_58_layer_call_fn_586306

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_586082o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name586302:&"
 
_user_specified_name586300:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
d
+__inference_dropout_38_layer_call_fn_586275

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_586070o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�(
�
!__inference__wrapped_model_586040
input_1J
8my_classifier_19_dense_57_matmul_readvariableop_resource: G
9my_classifier_19_dense_57_biasadd_readvariableop_resource: J
8my_classifier_19_dense_58_matmul_readvariableop_resource:  G
9my_classifier_19_dense_58_biasadd_readvariableop_resource: J
8my_classifier_19_dense_59_matmul_readvariableop_resource: G
9my_classifier_19_dense_59_biasadd_readvariableop_resource:
identity��0my_classifier_19/dense_57/BiasAdd/ReadVariableOp�/my_classifier_19/dense_57/MatMul/ReadVariableOp�0my_classifier_19/dense_58/BiasAdd/ReadVariableOp�/my_classifier_19/dense_58/MatMul/ReadVariableOp�0my_classifier_19/dense_59/BiasAdd/ReadVariableOp�/my_classifier_19/dense_59/MatMul/ReadVariableOp�
/my_classifier_19/dense_57/MatMul/ReadVariableOpReadVariableOp8my_classifier_19_dense_57_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
 my_classifier_19/dense_57/MatMulMatMulinput_17my_classifier_19/dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
0my_classifier_19/dense_57/BiasAdd/ReadVariableOpReadVariableOp9my_classifier_19_dense_57_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
!my_classifier_19/dense_57/BiasAddBiasAdd*my_classifier_19/dense_57/MatMul:product:08my_classifier_19/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
my_classifier_19/dense_57/ReluRelu*my_classifier_19/dense_57/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
$my_classifier_19/dropout_38/IdentityIdentity,my_classifier_19/dense_57/Relu:activations:0*
T0*'
_output_shapes
:��������� �
/my_classifier_19/dense_58/MatMul/ReadVariableOpReadVariableOp8my_classifier_19_dense_58_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
 my_classifier_19/dense_58/MatMulMatMul-my_classifier_19/dropout_38/Identity:output:07my_classifier_19/dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
0my_classifier_19/dense_58/BiasAdd/ReadVariableOpReadVariableOp9my_classifier_19_dense_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
!my_classifier_19/dense_58/BiasAddBiasAdd*my_classifier_19/dense_58/MatMul:product:08my_classifier_19/dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
my_classifier_19/dense_58/ReluRelu*my_classifier_19/dense_58/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
$my_classifier_19/dropout_39/IdentityIdentity,my_classifier_19/dense_58/Relu:activations:0*
T0*'
_output_shapes
:��������� �
/my_classifier_19/dense_59/MatMul/ReadVariableOpReadVariableOp8my_classifier_19_dense_59_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
 my_classifier_19/dense_59/MatMulMatMul-my_classifier_19/dropout_39/Identity:output:07my_classifier_19/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0my_classifier_19/dense_59/BiasAdd/ReadVariableOpReadVariableOp9my_classifier_19_dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!my_classifier_19/dense_59/BiasAddBiasAdd*my_classifier_19/dense_59/MatMul:product:08my_classifier_19/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!my_classifier_19/dense_59/SigmoidSigmoid*my_classifier_19/dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%my_classifier_19/dense_59/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp1^my_classifier_19/dense_57/BiasAdd/ReadVariableOp0^my_classifier_19/dense_57/MatMul/ReadVariableOp1^my_classifier_19/dense_58/BiasAdd/ReadVariableOp0^my_classifier_19/dense_58/MatMul/ReadVariableOp1^my_classifier_19/dense_59/BiasAdd/ReadVariableOp0^my_classifier_19/dense_59/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2d
0my_classifier_19/dense_57/BiasAdd/ReadVariableOp0my_classifier_19/dense_57/BiasAdd/ReadVariableOp2b
/my_classifier_19/dense_57/MatMul/ReadVariableOp/my_classifier_19/dense_57/MatMul/ReadVariableOp2d
0my_classifier_19/dense_58/BiasAdd/ReadVariableOp0my_classifier_19/dense_58/BiasAdd/ReadVariableOp2b
/my_classifier_19/dense_58/MatMul/ReadVariableOp/my_classifier_19/dense_58/MatMul/ReadVariableOp2d
0my_classifier_19/dense_59/BiasAdd/ReadVariableOp0my_classifier_19/dense_59/BiasAdd/ReadVariableOp2b
/my_classifier_19/dense_59/MatMul/ReadVariableOp/my_classifier_19/dense_59/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�g
�
"__inference__traced_restore_586593
file_prefix2
 assignvariableop_dense_57_kernel: .
 assignvariableop_1_dense_57_bias: 4
"assignvariableop_2_dense_58_kernel:  .
 assignvariableop_3_dense_58_bias: 4
"assignvariableop_4_dense_59_kernel: .
 assignvariableop_5_dense_59_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: ;
)assignvariableop_8_adam_m_dense_57_kernel: ;
)assignvariableop_9_adam_v_dense_57_kernel: 6
(assignvariableop_10_adam_m_dense_57_bias: 6
(assignvariableop_11_adam_v_dense_57_bias: <
*assignvariableop_12_adam_m_dense_58_kernel:  <
*assignvariableop_13_adam_v_dense_58_kernel:  6
(assignvariableop_14_adam_m_dense_58_bias: 6
(assignvariableop_15_adam_v_dense_58_bias: <
*assignvariableop_16_adam_m_dense_59_kernel: <
*assignvariableop_17_adam_v_dense_59_kernel: 6
(assignvariableop_18_adam_m_dense_59_bias:6
(assignvariableop_19_adam_v_dense_59_bias:#
assignvariableop_20_total: #
assignvariableop_21_count: 
identity_23��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_57_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_57_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_58_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_58_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_59_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_59_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp)assignvariableop_8_adam_m_dense_57_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp)assignvariableop_9_adam_v_dense_57_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp(assignvariableop_10_adam_m_dense_57_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp(assignvariableop_11_adam_v_dense_57_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adam_m_dense_58_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_v_dense_58_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_m_dense_58_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_v_dense_58_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_m_dense_59_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_v_dense_59_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_m_dense_59_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_v_dense_59_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_23Identity_23:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%!

_user_specified_namecount:%!

_user_specified_nametotal:40
.
_user_specified_nameAdam/v/dense_59/bias:40
.
_user_specified_nameAdam/m/dense_59/bias:62
0
_user_specified_nameAdam/v/dense_59/kernel:62
0
_user_specified_nameAdam/m/dense_59/kernel:40
.
_user_specified_nameAdam/v/dense_58/bias:40
.
_user_specified_nameAdam/m/dense_58/bias:62
0
_user_specified_nameAdam/v/dense_58/kernel:62
0
_user_specified_nameAdam/m/dense_58/kernel:40
.
_user_specified_nameAdam/v/dense_57/bias:40
.
_user_specified_nameAdam/m/dense_57/bias:6
2
0
_user_specified_nameAdam/v/dense_57/kernel:6	2
0
_user_specified_nameAdam/m/dense_57/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_59/bias:/+
)
_user_specified_namedense_59/kernel:-)
'
_user_specified_namedense_58/bias:/+
)
_user_specified_namedense_58/kernel:-)
'
_user_specified_namedense_57/bias:/+
)
_user_specified_namedense_57/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
G
+__inference_dropout_38_layer_call_fn_586280

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_586130`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
G
+__inference_dropout_39_layer_call_fn_586327

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_586141`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
1__inference_my_classifier_19_layer_call_fn_586183
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_my_classifier_19_layer_call_and_return_conditional_losses_586149o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name586179:&"
 
_user_specified_name586177:&"
 
_user_specified_name586175:&"
 
_user_specified_name586173:&"
 
_user_specified_name586171:&"
 
_user_specified_name586169:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
��
�
__inference__traced_save_586518
file_prefix8
&read_disablecopyonread_dense_57_kernel: 4
&read_1_disablecopyonread_dense_57_bias: :
(read_2_disablecopyonread_dense_58_kernel:  4
&read_3_disablecopyonread_dense_58_bias: :
(read_4_disablecopyonread_dense_59_kernel: 4
&read_5_disablecopyonread_dense_59_bias:,
"read_6_disablecopyonread_iteration:	 0
&read_7_disablecopyonread_learning_rate: A
/read_8_disablecopyonread_adam_m_dense_57_kernel: A
/read_9_disablecopyonread_adam_v_dense_57_kernel: <
.read_10_disablecopyonread_adam_m_dense_57_bias: <
.read_11_disablecopyonread_adam_v_dense_57_bias: B
0read_12_disablecopyonread_adam_m_dense_58_kernel:  B
0read_13_disablecopyonread_adam_v_dense_58_kernel:  <
.read_14_disablecopyonread_adam_m_dense_58_bias: <
.read_15_disablecopyonread_adam_v_dense_58_bias: B
0read_16_disablecopyonread_adam_m_dense_59_kernel: B
0read_17_disablecopyonread_adam_v_dense_59_kernel: <
.read_18_disablecopyonread_adam_m_dense_59_bias:<
.read_19_disablecopyonread_adam_v_dense_59_bias:)
read_20_disablecopyonread_total: )
read_21_disablecopyonread_count: 
savev2_const
identity_45��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_57_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_57_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

: z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_57_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_57_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_58_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_58_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:  z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_58_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_58_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_59_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_59_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

: z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_59_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_59_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_iteration^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_learning_rate^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead/read_8_disablecopyonread_adam_m_dense_57_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp/read_8_disablecopyonread_adam_m_dense_57_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_9/DisableCopyOnReadDisableCopyOnRead/read_9_disablecopyonread_adam_v_dense_57_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp/read_9_disablecopyonread_adam_v_dense_57_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_10/DisableCopyOnReadDisableCopyOnRead.read_10_disablecopyonread_adam_m_dense_57_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp.read_10_disablecopyonread_adam_m_dense_57_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_11/DisableCopyOnReadDisableCopyOnRead.read_11_disablecopyonread_adam_v_dense_57_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp.read_11_disablecopyonread_adam_v_dense_57_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnRead0read_12_disablecopyonread_adam_m_dense_58_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp0read_12_disablecopyonread_adam_m_dense_58_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_13/DisableCopyOnReadDisableCopyOnRead0read_13_disablecopyonread_adam_v_dense_58_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp0read_13_disablecopyonread_adam_v_dense_58_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_14/DisableCopyOnReadDisableCopyOnRead.read_14_disablecopyonread_adam_m_dense_58_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp.read_14_disablecopyonread_adam_m_dense_58_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_15/DisableCopyOnReadDisableCopyOnRead.read_15_disablecopyonread_adam_v_dense_58_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp.read_15_disablecopyonread_adam_v_dense_58_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_16/DisableCopyOnReadDisableCopyOnRead0read_16_disablecopyonread_adam_m_dense_59_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp0read_16_disablecopyonread_adam_m_dense_59_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_17/DisableCopyOnReadDisableCopyOnRead0read_17_disablecopyonread_adam_v_dense_59_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp0read_17_disablecopyonread_adam_v_dense_59_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_18/DisableCopyOnReadDisableCopyOnRead.read_18_disablecopyonread_adam_m_dense_59_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp.read_18_disablecopyonread_adam_m_dense_59_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnRead.read_19_disablecopyonread_adam_v_dense_59_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp.read_19_disablecopyonread_adam_v_dense_59_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_20/DisableCopyOnReadDisableCopyOnReadread_20_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOpread_20_disablecopyonread_total^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_21/DisableCopyOnReadDisableCopyOnReadread_21_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOpread_21_disablecopyonread_count^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: �	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *%
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_44Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_45IdentityIdentity_44:output:0^NoOp*
T0*
_output_shapes
: �	
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_45Identity_45:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:%!

_user_specified_namecount:%!

_user_specified_nametotal:40
.
_user_specified_nameAdam/v/dense_59/bias:40
.
_user_specified_nameAdam/m/dense_59/bias:62
0
_user_specified_nameAdam/v/dense_59/kernel:62
0
_user_specified_nameAdam/m/dense_59/kernel:40
.
_user_specified_nameAdam/v/dense_58/bias:40
.
_user_specified_nameAdam/m/dense_58/bias:62
0
_user_specified_nameAdam/v/dense_58/kernel:62
0
_user_specified_nameAdam/m/dense_58/kernel:40
.
_user_specified_nameAdam/v/dense_57/bias:40
.
_user_specified_nameAdam/m/dense_57/bias:6
2
0
_user_specified_nameAdam/v/dense_57/kernel:6	2
0
_user_specified_nameAdam/m/dense_57/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_59/bias:/+
)
_user_specified_namedense_59/kernel:-)
'
_user_specified_namedense_58/bias:/+
)
_user_specified_namedense_58/kernel:-)
'
_user_specified_namedense_57/bias:/+
)
_user_specified_namedense_57/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
D__inference_dense_57_layer_call_and_return_conditional_losses_586270

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
F__inference_dropout_38_layer_call_and_return_conditional_losses_586297

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
d
F__inference_dropout_39_layer_call_and_return_conditional_losses_586141

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
D__inference_dense_57_layer_call_and_return_conditional_losses_586053

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_dense_59_layer_call_fn_586353

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_586111o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name586349:&"
 
_user_specified_name586347:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
d
+__inference_dropout_39_layer_call_fn_586322

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_586099o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
)__inference_dense_57_layer_call_fn_586259

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_586053o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name586255:&"
 
_user_specified_name586253:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1
	dropout1


dense2
dropout2
output_layer
	optimizer

metric

signatures"
_tf_keras_model
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_12�
1__inference_my_classifier_19_layer_call_fn_586166
1__inference_my_classifier_19_layer_call_fn_586183�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1
�
trace_0
trace_12�
L__inference_my_classifier_19_layer_call_and_return_conditional_losses_586118
L__inference_my_classifier_19_layer_call_and_return_conditional_losses_586149�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1
�B�
!__inference__wrapped_model_586040input_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_random_generator"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8_random_generator"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
?
_variables
@_iterations
A_learning_rate
B_index_dict
C
_momentums
D_velocities
E_update_step_xla"
experimentalOptimizer
 "
trackable_list_wrapper
,
Fserving_default"
signature_map
!: 2dense_57/kernel
: 2dense_57/bias
!:  2dense_58/kernel
: 2dense_58/bias
!: 2dense_59/kernel
:2dense_59/bias
 "
trackable_list_wrapper
C
0
	1

2
3
4"
trackable_list_wrapper
'
G0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_my_classifier_19_layer_call_fn_586166input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
1__inference_my_classifier_19_layer_call_fn_586183input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
L__inference_my_classifier_19_layer_call_and_return_conditional_losses_586118input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
L__inference_my_classifier_19_layer_call_and_return_conditional_losses_586149input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
Mtrace_02�
)__inference_dense_57_layer_call_fn_586259�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zMtrace_0
�
Ntrace_02�
D__inference_dense_57_layer_call_and_return_conditional_losses_586270�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zNtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
Ttrace_0
Utrace_12�
+__inference_dropout_38_layer_call_fn_586275
+__inference_dropout_38_layer_call_fn_586280�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zTtrace_0zUtrace_1
�
Vtrace_0
Wtrace_12�
F__inference_dropout_38_layer_call_and_return_conditional_losses_586292
F__inference_dropout_38_layer_call_and_return_conditional_losses_586297�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zVtrace_0zWtrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
]trace_02�
)__inference_dense_58_layer_call_fn_586306�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z]trace_0
�
^trace_02�
D__inference_dense_58_layer_call_and_return_conditional_losses_586317�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z^trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
dtrace_0
etrace_12�
+__inference_dropout_39_layer_call_fn_586322
+__inference_dropout_39_layer_call_fn_586327�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0zetrace_1
�
ftrace_0
gtrace_12�
F__inference_dropout_39_layer_call_and_return_conditional_losses_586339
F__inference_dropout_39_layer_call_and_return_conditional_losses_586344�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zftrace_0zgtrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
mtrace_02�
)__inference_dense_59_layer_call_fn_586353�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zmtrace_0
�
ntrace_02�
D__inference_dense_59_layer_call_and_return_conditional_losses_586364�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zntrace_0
~
@0
o1
p2
q3
r4
s5
t6
u7
v8
w9
x10
y11
z12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
J
o0
q1
s2
u3
w4
y5"
trackable_list_wrapper
J
p0
r1
t2
v3
x4
z5"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_586250input_1"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
	jinput_1
kwonlydefaults
 
annotations� *
 
N
{	variables
|	keras_api
	}total
	~count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_57_layer_call_fn_586259inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_57_layer_call_and_return_conditional_losses_586270inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_38_layer_call_fn_586275inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_38_layer_call_fn_586280inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_38_layer_call_and_return_conditional_losses_586292inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_38_layer_call_and_return_conditional_losses_586297inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_58_layer_call_fn_586306inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_58_layer_call_and_return_conditional_losses_586317inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_39_layer_call_fn_586322inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_39_layer_call_fn_586327inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_39_layer_call_and_return_conditional_losses_586339inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_39_layer_call_and_return_conditional_losses_586344inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_59_layer_call_fn_586353inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_59_layer_call_and_return_conditional_losses_586364inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
&:$ 2Adam/m/dense_57/kernel
&:$ 2Adam/v/dense_57/kernel
 : 2Adam/m/dense_57/bias
 : 2Adam/v/dense_57/bias
&:$  2Adam/m/dense_58/kernel
&:$  2Adam/v/dense_58/kernel
 : 2Adam/m/dense_58/bias
 : 2Adam/v/dense_58/bias
&:$ 2Adam/m/dense_59/kernel
&:$ 2Adam/v/dense_59/kernel
 :2Adam/m/dense_59/bias
 :2Adam/v/dense_59/bias
.
}0
~1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
:  (2total
:  (2count�
!__inference__wrapped_model_586040o0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
D__inference_dense_57_layer_call_and_return_conditional_losses_586270c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0��������� 
� �
)__inference_dense_57_layer_call_fn_586259X/�,
%�"
 �
inputs���������
� "!�
unknown��������� �
D__inference_dense_58_layer_call_and_return_conditional_losses_586317c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
)__inference_dense_58_layer_call_fn_586306X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
D__inference_dense_59_layer_call_and_return_conditional_losses_586364c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
)__inference_dense_59_layer_call_fn_586353X/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
F__inference_dropout_38_layer_call_and_return_conditional_losses_586292c3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
F__inference_dropout_38_layer_call_and_return_conditional_losses_586297c3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
+__inference_dropout_38_layer_call_fn_586275X3�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
+__inference_dropout_38_layer_call_fn_586280X3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� �
F__inference_dropout_39_layer_call_and_return_conditional_losses_586339c3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
F__inference_dropout_39_layer_call_and_return_conditional_losses_586344c3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
+__inference_dropout_39_layer_call_fn_586322X3�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
+__inference_dropout_39_layer_call_fn_586327X3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� �
L__inference_my_classifier_19_layer_call_and_return_conditional_losses_586118x@�=
&�#
!�
input_1���������
�

trainingp",�)
"�
tensor_0���������
� �
L__inference_my_classifier_19_layer_call_and_return_conditional_losses_586149x@�=
&�#
!�
input_1���������
�

trainingp ",�)
"�
tensor_0���������
� �
1__inference_my_classifier_19_layer_call_fn_586166m@�=
&�#
!�
input_1���������
�

trainingp"!�
unknown����������
1__inference_my_classifier_19_layer_call_fn_586183m@�=
&�#
!�
input_1���������
�

trainingp "!�
unknown����������
$__inference_signature_wrapper_586250z;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1���������