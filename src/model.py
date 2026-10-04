"""
Xceptional Vision - Model Architecture Module
=============================================
Builds the transfer learning network using the pre-trained Xception backbone with ImageNet weights
and a customized classification head for lung cancer tissue pattern classification.

DISCLAIMER:
-----------
This model and codebase are developed strictly for educational and research purposes
in computer vision and biomedical image analysis. This software is NOT a certified medical
diagnostic tool and must NOT be used for clinical decision-making or diagnosis.
"""

import os
import sys
from typing import Tuple, List, Optional, Union, Dict, Any

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras.applications import Xception


# Default standard input dimensions for Xception
DEFAULT_INPUT_SHAPE: Tuple[int, int, int] = (299, 299, 3)
DEFAULT_NUM_CLASSES: int = 4
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_DROPOUT_RATE: float = 0.4
DEFAULT_DENSE_UNITS: int = 256


def build_xceptional_model(
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    freeze_backbone: bool = True,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    dense_units: int = DEFAULT_DENSE_UNITS,
    model_name: str = "XceptionalVision_Classifier"
) -> tf.keras.Model:
    """
    Constructs the Xception-based deep learning architecture:
      1. Pre-trained Xception feature extractor (ImageNet weights, top layer excluded).
      2. Backbone freezing to preserve general visual representations during initial training.
      3. GlobalAveragePooling2D layer to collapse 2D spatial feature maps into a 1D vector.
      4. BatchNormalization layer to stabilize feature distribution and accelerate convergence.
      5. Fully connected Dense layer (ReLU activation) to learn domain-specific patterns.
      6. Dropout regularization layer to combat overfitting.
      7. Final classification Dense layer (Softmax for multiclass, Sigmoid for binary).

    Args:
        input_shape (Tuple[int, int, int]): Dimensions of input images (Height, Width, Channels). Default: (299, 299, 3).
        num_classes (int): Number of target output classes. Default: 4.
        freeze_backbone (bool): If True, freezes all weights in the Xception base. Default: True.
        dropout_rate (float): Fraction of neurons randomly dropped during training. Default: 0.4.
        dense_units (int): Number of neurons in the intermediate dense layer. Default: 256.
        model_name (str): Identifier name for the generated Keras model. Default: "XceptionalVision_Classifier".

    Returns:
        tf.keras.Model: The assembled Keras Functional Model ready for compilation and training.
    """
    # 1. Define Model Input
    inputs = layers.Input(shape=input_shape, name="input_image")

    # 2. Load Pre-trained Xception Backbone (ImageNet weights, exclude 1000-class top)
    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model._name = "xception_backbone"

    # 3. Freeze or Unfreeze the Backbone
    base_model.trainable = not freeze_backbone

    # 4. Pass inputs through the backbone
    # Note: training=False ensures BatchNormalization layers inside the frozen base stay in inference mode
    x = base_model(inputs, training=False)

    # 5. Custom Classification Head
    # 5a. Global Average Pooling 2D: spatial reduction (H x W x C -> 1 x 1 x C -> C)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)

    # 5b. Batch Normalization: normalize activations across the mini-batch
    x = layers.BatchNormalization(name="batch_normalization")(x)

    # 5c. Fully-Connected Dense Layer: domain feature mapping
    x = layers.Dense(dense_units, activation="relu", name="dense_features")(x)

    # 5d. Dropout Regularization: randomly zero out units to prevent co-adaptation/overfitting
    x = layers.Dropout(dropout_rate, name="dropout_regularization")(x)

    # 5e. Final Classification Layer: determine class probabilities
    if num_classes == 1:
        # Single-unit output for binary classification (probability of positive class)
        outputs = layers.Dense(1, activation="sigmoid", name="classification_output")(x)
    else:
        # Multi-unit output for multiclass classification (probability distribution summing to 1.0)
        outputs = layers.Dense(num_classes, activation="softmax", name="classification_output")(x)

    # 6. Instantiate Functional Model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    loss: Optional[Union[str, tf.keras.losses.Loss]] = None,
    metrics_list: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None
) -> tf.keras.Model:
    """
    Compiles the Keras model with Adam optimizer, appropriate loss function, and tracking metrics.

    Args:
        model (tf.keras.Model): The uncompiled Keras model.
        learning_rate (float): Learning rate for the Adam optimizer. Default: 1e-4.
        num_classes (int): Number of target output classes (used to determine default loss if not provided).
        loss (Optional[Union[str, tf.keras.losses.Loss]]): Loss function.
            If None, defaults to:
              - 'binary_crossentropy' when num_classes == 1
              - 'categorical_crossentropy' when num_classes > 1
        metrics_list (Optional[List[Union[str, tf.keras.metrics.Metric]]]): Evaluation metrics to track.
            If None, tracks accuracy, precision, recall, and AUC.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    # 1. Configure Optimizer
    optimizer = optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        name="adam_optimizer"
    )

    # 2. Select Appropriate Loss Function
    if loss is None:
        if num_classes == 1:
            loss = losses.BinaryCrossentropy(name="binary_crossentropy")
        else:
            loss = losses.CategoricalCrossentropy(name="categorical_crossentropy")

    # 3. Configure Default Metrics
    if metrics_list is None:
        metrics_list = [
            metrics.CategoricalAccuracy(name="accuracy") if num_classes > 1 else metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc")
        ]

    # 4. Compile Model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics_list
    )

    return model


def unfreeze_backbone_for_fine_tuning(
    model: tf.keras.Model,
    num_unfreeze_layers: int = 20,
    fine_tuning_lr: float = 1e-5,
    num_classes: int = DEFAULT_NUM_CLASSES
) -> tf.keras.Model:
    """
    Unfreezes the top N layers of the Xception backbone for Phase 2 fine-tuning.
    Recompiles the model with a lower learning rate to gently adapt high-level visual features.

    Args:
        model (tf.keras.Model): The trained baseline model.
        num_unfreeze_layers (int): Number of layers from the end of the backbone to make trainable.
        fine_tuning_lr (float): Reduced learning rate for fine-tuning. Default: 1e-5.
        num_classes (int): Number of dataset classes for metric configuration.

    Returns:
        tf.keras.Model: Recompiled model ready for fine-tuning.
    """
    # Locate backbone layer
    backbone = None
    for layer in model.layers:
        if "xception" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise ValueError("Could not find Xception backbone layer in the model.")

    # Enable backbone training
    backbone.trainable = True

    # Freeze all layers except the last `num_unfreeze_layers`
    total_layers = len(backbone.layers)
    freeze_until = max(0, total_layers - num_unfreeze_layers)

    for i, layer in enumerate(backbone.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            # Keep BatchNormalization frozen during fine-tuning for stability
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    # Recompile with smaller learning rate
    recompiled_model = compile_model(
        model=model,
        learning_rate=fine_tuning_lr,
        num_classes=num_classes
    )

    return recompiled_model


def get_layer_explanations() -> Dict[str, Dict[str, str]]:
    """
    Provides beginner-friendly, educational explanations for each major layer
    in the Xception transfer learning architecture.

    Returns:
        Dict[str, Dict[str, str]]: Structured explanations of every major layer.
    """
    return {
        "1. Input Layer (layers.Input)": {
            "Purpose": "Defines the shape and format of images fed into the network.",
            "Explanation": (
                "Think of this as the entrance gate. It tells TensorFlow that every incoming CT image "
                "must have a resolution of 299x299 pixels and 3 color channels (RGB). It prepares the "
                "data pipeline without modifying the image values."
            ),
            "Dimensions": "(None, 299, 299, 3)"
        },
        "2. Xception Backbone (Pre-trained Feature Extractor)": {
            "Purpose": "Extracts rich, hierarchical visual features (edges, textures, shapes, anatomical tissue patterns).",
            "Explanation": (
                "Xception stands for 'Extreme Inception'. It has been pre-trained on over 1.4 million images "
                "(ImageNet) to recognize fundamental visual patterns. It uses 'Depthwise Separable Convolutions', "
                "which look at spatial patterns (height x width) and channel patterns (color/depth) separately. "
                "This makes the model dramatically faster and more parameter-efficient than traditional CNNs while "
                "capturing subtle tissue density changes in CT scans."
            ),
            "Dimensions": "Output feature map: (None, 10, 10, 2048)"
        },
        "3. Frozen Weights (base_model.trainable = False)": {
            "Purpose": "Preserves foundational visual knowledge and protects against destructive updates.",
            "Explanation": (
                "When we freeze the backbone, its 20+ million weights are locked like a sealed encyclopedia. "
                "The randomly initialized top classification layers won't ruin the pre-trained feature detectors "
                "during the initial training epochs. Only our custom classification head learns at first."
            ),
            "Dimensions": "N/A (Parameter State: Non-trainable)"
        },
        "4. GlobalAveragePooling2D (layers.GlobalAveragePooling2D)": {
            "Purpose": "Reduces 2D spatial feature maps into a single 1D feature summary vector per channel.",
            "Explanation": (
                "The backbone outputs a 10x10 grid with 2,048 feature maps (a total of 204,800 numbers per image). "
                "GlobalAveragePooling2D calculates the average value across each 10x10 map, condensing each map into "
                "a single representative number. This yields a clean vector of 2,048 numbers, reducing overfitting "
                "and eliminating millions of unnecessary parameters compared to a traditional Flatten layer."
            ),
            "Dimensions": "(None, 2048)"
        },
        "5. BatchNormalization (layers.BatchNormalization)": {
            "Purpose": "Normalizes activations to have zero mean and unit variance across mini-batches.",
            "Explanation": (
                "As data flows through deep networks, the distribution of layer activations shifts (internal covariate shift). "
                "BatchNormalization rescales and re-centers the feature values for each batch. This stabilizes the learning "
                "process, allows higher learning rates, and provides subtle regularization."
            ),
            "Dimensions": "(None, 2048)"
        },
        "6. Dense Hidden Layer (layers.Dense with ReLU)": {
            "Purpose": "Combines extracted visual features to learn high-level task-specific patterns.",
            "Explanation": (
                "This fully connected layer has 256 artificial neurons. Each neuron looks at all 2,048 pooled features "
                "and learns specialized combinations relevant to chest CT scans (e.g., distinguishing smooth lung borders "
                "from irregular lobulated nodules). The ReLU (Rectified Linear Unit) activation function introduces non-linearity, "
                "enabling the network to model complex mathematical relationships."
            ),
            "Dimensions": "(None, 256)"
        },
        "7. Dropout Regularization (layers.Dropout, rate=0.4)": {
            "Purpose": "Prevents neural co-adaptation and memorization (overfitting).",
            "Explanation": (
                "During each training step, this layer randomly 'turns off' 40% of the neurons in the dense layer. "
                "This forces the network to find robust, distributed representations rather than relying on any single "
                "dominant neuron or memorizing training samples."
            ),
            "Dimensions": "(None, 256)"
        },
        "8. Final Classification Layer (layers.Dense with Softmax/Sigmoid)": {
            "Purpose": "Outputs probability scores for each target class.",
            "Explanation": (
                "For multiclass classification (4 classes: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal), "
                "this layer has 4 neurons with the Softmax activation function. Softmax converts raw output scores into a valid "
                "probability distribution where all 4 scores are between 0.0 and 1.0 and sum to exactly 1.0 (100%)."
            ),
            "Dimensions": "(None, 4) [or (None, 1) for binary]"
        }
    }


def display_model_summary(model: tf.keras.Model, show_explanations: bool = True) -> None:
    """
    Displays the Keras model.summary() along with beginner-friendly layer explanations.

    Args:
        model (tf.keras.Model): The constructed Keras model.
        show_explanations (bool): Whether to also print layer explanations. Default: True.
    """
    print("=" * 80)
    print("XCEPTIONAL VISION - MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)
    model.summary(expand_nested=False, line_length=80)
    print("=" * 80)

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    total_count = trainable_count + non_trainable_count

    print(f"\nParameter Breakdown:")
    print(f"  • Total Parameters:         {total_count:,}")
    print(f"  • Trainable Parameters:     {trainable_count:,} (Custom Head)")
    print(f"  • Non-trainable Parameters: {non_trainable_count:,} (Frozen Xception Backbone)")
    print("-" * 80)

    if show_explanations:
        print("\n" + "=" * 80)
        print("BEGINNER-FRIENDLY LAYER-BY-LAYER EXPLANATION")
        print("=" * 80)
        explanations = get_layer_explanations()
        for layer_name, details in explanations.items():
            print(f"\n[{layer_name}]")
            print(f"  • Role:        {details['Purpose']}")
            print(f"  • Dimensions:  {details['Dimensions']}")
            print(f"  • Explanation: {details['Explanation']}")
        print("\n" + "=" * 80)
        print("DISCLAIMER: Educational/Research project. Not for clinical diagnosis.")
        print("=" * 80)


if __name__ == "__main__":
    print("\n[Step 1/3] Instantiating Xception Transfer Learning Architecture...")
    xceptional_model = build_xceptional_model(
        input_shape=(299, 299, 3),
        num_classes=4,
        freeze_backbone=True,
        dropout_rate=0.4,
        dense_units=256
    )

    print("\n[Step 2/3] Compiling Model with Adam Optimizer & Categorical Crossentropy...")
    compiled_model = compile_model(
        model=xceptional_model,
        learning_rate=1e-4,
        num_classes=4
    )

    print("\n[Step 3/3] Displaying Model Summary & Layer Explanations...\n")
    display_model_summary(compiled_model, show_explanations=True)
