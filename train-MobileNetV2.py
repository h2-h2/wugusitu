import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# =============================================================================
# 第一阶段：构建模型并冻结预训练层，只训练分类头
# =============================================================================

# 使用 ImageNet 预训练权重（首次运行自动下载）
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# 分类头：使用L2正则化防止过拟合，BatchNormalization加速收敛
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu',
          kernel_regularizer=regularizers.l2(0.001))(x)  # L2正则化，防止过拟合
x = BatchNormalization()(x)  # 批归一化，加速收敛
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',
          kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 冻结预训练层
base_model.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =============================================================================
# 数据增强配置
# =============================================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

# =============================================================================
# 第一阶段回调（独立于第二阶段）
# =============================================================================

phase1_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 第一阶段最佳模型保存到独立文件，不会被第二阶段覆盖
phase1_checkpoint = ModelCheckpoint(
    'models/mobilenetv2_grains_phase1_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

phase1_reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# =============================================================================
# 第一阶段训练（只训练分类头）
# =============================================================================

print("=" * 60)
print("第一阶段训练：只训练新添加的分类头")
print("=" * 60)

history_phase1 = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[phase1_early_stopping, phase1_checkpoint, phase1_reduce_lr],
    verbose=1
)

# =============================================================================
# 第二阶段：解冻顶层卷积 + 小学习率微调（小样本安全微调）
# =============================================================================

print("\n" + "=" * 60)
print("第二阶段训练：解冻最后5层卷积，小学习率微调")
print("=" * 60)

# 只解冻最后5层，前面全部冻结（MobileNetV2层数少，更保守）
base_model.trainable = True
for layer in base_model.layers[:-5]:
    layer.trainable = False

# 极小学习率，防止破坏预训练特征
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 第二阶段回调
phase2_early_stopping = EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
)

phase2_checkpoint = ModelCheckpoint(
    'models/mobilenetv2_grains.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

phase2_reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1
)

history_phase2 = model.fit(
    train_generator,
    epochs=60,
    initial_epoch=history_phase1.epoch[-1] + 1,
    validation_data=val_generator,
    callbacks=[phase2_early_stopping, phase2_checkpoint, phase2_reduce_lr],
    verbose=1
)

print("\n" + "=" * 60)
print("训练完成！")
print(f"第一阶段最佳模型：models/mobilenetv2_grains_phase1_best.h5")
print(f"最终模型（微调后）：models/mobilenetv2_grains.h5")
print("=" * 60)