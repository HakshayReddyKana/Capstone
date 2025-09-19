# DetDSHAP YOLOv8 Pruning: A Real Implementation Plan (Version 3)

This document outlines the step-by-step engineering process to replicate the DetDSHAP pruning methodology on a YOLOv8 model. This revised plan addresses critical flaws in previous versions and commits to a more robust, non-simulated implementation.

### **Guiding Principles**

*   **Perfection and Verification:** After each phase, all outputs and results will be thoroughly reviewed and validated for correctness and perfection. Only after this rigorous check is complete will the work be committed.
*   **Sequential Progress:** We will only proceed to the next phase after the current phase is 100% complete, validated, and committed.

### **Phase 1: Foundational Tooling (Hierarchical Graph Parsing)**

*   **Objective:** To deeply understand the model's complex, non-sequential, and hierarchical architecture.
*   **Action:** Enhance the `build_dependency_graph` function to be hierarchical and add validation.
*   **Method:**
    1.  The parser must not treat complex modules like `C2f` as black boxes.
    2.  It will recursively parse the sub-modules within `C2f` and other composite layers.
    3.  The final graph will represent the full, detailed computational path, including internal connections within modules.
    4.  **Graph Validation:** Implement a `validate_graph` function. This function will programmatically trace the forward pass by traversing the generated graph and ensure its structure aligns with the actual model's tensor flow, confirming correctness before proceeding.
*   **Status:** Not Started.
*   **Checkpoint:**
    - After validating the hierarchical graph parser, run:
      ```sh
      git add .
      git commit -m "[DetDSHAP] Phase 1 complete: Hierarchical model dependency graph parser implemented and validated."
      ```

### **Phase 2: Graph-Based Explainer (The DetDSHAP Algorithm)**

*   **Objective:** To implement the core of the paper's contribution: a custom backward pass that correctly calculates SHAP-based relevance.
*   **Action:** Re-implement the `DetDSHAP.explain` method to be fully graph-aware.
*   **Method:**
    1.  Perform a forward pass, using hooks to capture the activation tensor of every layer in the hierarchical graph.
    2.  **Relevance Initialization:**
        *   Given a target bounding box, identify the best-matching anchor and responsible grid cell on the corresponding feature map scale.
        *   Initialize relevance as `1.0` only at the specific tensor indices (batch, anchor, grid_y, grid_x, class_id) corresponding to this best match. All other elements of the output tensor will have a relevance of `0`.
    3.  Perform a topological traversal of the hierarchical dependency graph in reverse order.
    4.  For each layer, apply a specific, well-defined relevance propagation rule:
        *   **`Concat` Layers:** Split relevance proportionally to the L1-norm of the activation contributed by each parent branch.
        *   **`C2f` Modules:** Traverse the internal sub-graph of the module, propagating relevance through its internal convolutions and splits.
        *   **`Detect` Head:** The backward pass must correctly traverse the internal convolutional layers of the `Detect` module, propagating relevance from the final output tensor back to its three input feature maps.
        *   **`Conv2d` & `SiLU` Layers:** Apply the specific LRP and custom derivative rules as defined in the paper.
*   **Outcome:**
    1.  A dictionary mapping every layer to its correctly calculated relevance tensor.
    2.  **Visualization:** For a sample image, generate and save a visual output showing the original image, the detected bounding box for the explained object, and the SHAP relevance map overlaid as a heatmap. This provides concrete visual proof of the explainer's function.
*   **Checkpoint:**
    - After the explainer produces correct, real SHAP maps and a visualization, run:
      ```sh
      git add .
      git commit -m "[DetDSHAP] Phase 2 complete: Graph-based DetDSHAP explainer implemented and visualized."
      ```

### **Phase 3: Real, Class-Balanced Importance Calculation**

*   **Objective:** To generate an unbiased, concrete importance score for every filter.
*   **Action:** Create a `calculate_real_filter_importance` function with a class-balanced sampling strategy.
*   **Method:**
    1.  **Address Class Imbalance:** Create batches from the validation set that ensure representation from all classes, especially rare ones.
    2.  For each image in the balanced batch, run the `DetDSHAP.explain` method.
    3.  Aggregate the absolute SHAP values across the balanced batch to get a final, unbiased importance score for every filter.
*   **Outcome:** A `pruning_plan` dictionary based on fair and accurate SHAP values.
*   **Checkpoint:**
    - After generating and verifying real, class-balanced filter importance scores, run:
      ```sh
      git add .
      git commit -m "[DetDSHAP] Phase 3 complete: Class-balanced filter importance calculation."
      ```

### **Phase 4: Graph-Based Model Surgery (Architecture Reconstruction)**

*   **Objective:** To physically remove the least important filters/layers and correctly reconstruct the model.
*   **Action:** Re-implement `apply_pruning_plan` to perform true model reconstruction.
*   **Method:**
    1.  **Pruning Granularity:** The framework will support both **filter-level** (removing individual filters) and **layer-level** (removing entire layers/modules) pruning, configurable via a parameter.
    2.  **Generate New Architecture:** Based on the pruning plan, programmatically generate a new YOLOv8 model definition `.yaml` file.
    3.  **Instantiate New Model:** Load this new `.yaml` file to create a new, smaller `torch.nn.Module`.
    4.  **Painstakingly Copy Weights:** Write a script that iterates through the original and new models. It will use the dependency graph to copy the weights of every *unpruned* element to its corresponding position in the new model.
    5.  **`Detect` Head Reconstruction:** The reconstruction process must be able to modify the `Detect` head's internal layers to adapt to changes in its input channel counts, ensuring the final model is valid.
*   **Outcome:** A new, smaller, and fully functional `torch.nn.Module` representing the pruned YOLOv8 model.
*   **Checkpoint:**
    - After reconstructing and validating the pruned model, run:
      ```sh
      git add .
      git commit -m "[DetDSHAP] Phase 4 complete: Model architecture reconstruction and weight copying."
      ```

### **Phase 5: Performance Recovery**

*   **Objective:** To regain accuracy lost during pruning.
*   **Action:** Implement a fine-tuning loop.
*   **Method:** Use the standard `ultralytics` training pipeline to fine-tune the newly created pruned model on the training dataset.
*   **Outcome:** A final, optimized, pruned model.
*   **Checkpoint:**
    - After fine-tuning, run:
      ```sh
      git add .
      git commit -m "[DetDSHAP] Phase 5 complete: Fine-tuning of pruned model."
      ```

### **Phase 6: Final Evaluation**

*   **Objective:** To provide a final, honest comparison.
*   **Action:** Create a final evaluation and comparison table.
*   **Method:** Run `model.val()` on both the original model and the final, fine-tuned pruned model. This command evaluates performance across the **entire validation dataset** as specified in the project's data configuration file.
*   **Outcome:** A final report table containing **real, measured metrics** (mAP@0.5, mAP@0.5-0.95, Parameters, FLOPs, F1, Recall) from the full validation run.
*   **Checkpoint:**
    - After generating the final evaluation, run:
      ```sh
      git add .
      git commit -m "[DetDSHAP] Phase 6 complete: Final evaluation and comparison."
      ```
