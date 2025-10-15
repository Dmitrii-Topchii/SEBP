import torch
import torch.nn as nn
from sparse_method_triton import SparseMethodLinearIntermediate

def run_gradient_test():
    """
    Compares the gradients of a standard nn.Linear layer against the custom
    SparseMethodLinearIntermediate layer to ensure the backward pass is correct.
    """
    # Test setup
    batch_size, in_features, out_features = 2, 4, 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping gradient test.")
        exit(0)

    # Create a standard linear layer
    linear_layer = nn.Linear(in_features, out_features).to(device)

    # Create our custom sparse linear layer and load the same weights
    sparse_layer = SparseMethodLinearIntermediate(in_features, out_features, layer_idx=0).to(device)
    sparse_layer.weight = nn.Parameter(linear_layer.weight.clone())
    sparse_layer.bias = nn.Parameter(linear_layer.bias.clone())

    # Create a random input tensor
    input_tensor = torch.randn(batch_size, in_features, device=device, requires_grad=True)

    # --- Standard Model ---
    output_standard = linear_layer(input_tensor)
    loss_standard = output_standard.sum()
    loss_standard.backward()

    # --- Sparse Model ---
    # The input tensor for the sparse model must be a fresh clone
    input_tensor_sparse = input_tensor.clone().detach().requires_grad_(True)
    output_sparse = sparse_layer(input_tensor_sparse)
    loss_sparse = output_sparse.sum()
    loss_sparse.backward()

    # --- Comparison ---
    print("--- Gradient Comparison ---")

    # Compare weight gradients
    grad_weight_standard = linear_layer.weight.grad
    grad_weight_sparse = sparse_layer.weight.grad
    weight_grad_diff = torch.allclose(grad_weight_standard, grad_weight_sparse)
    print(f"Weight Gradients Match: {weight_grad_diff}")

    # Compare bias gradients
    grad_bias_standard = linear_layer.bias.grad
    grad_bias_sparse = sparse_layer.bias.grad
    bias_grad_diff = torch.allclose(grad_bias_standard, grad_bias_sparse)
    print(f"Bias Gradients Match: {bias_grad_diff}")

    # Compare input gradients
    grad_input_standard = input_tensor.grad
    grad_input_sparse = input_tensor_sparse.grad
    input_grad_diff = torch.allclose(grad_input_standard, grad_input_sparse)
    print(f"Input Gradients Match: {input_grad_diff}")

    if not all([weight_grad_diff, bias_grad_diff, input_grad_diff]):
        print("\nGradient check FAILED.")
        # Optional: Print gradients for debugging
        # print("\nStandard Weight Grad:\n", grad_weight_standard)
        # print("Sparse Weight Grad:\n", grad_weight_sparse)
        # print("\nStandard Input Grad:\n", grad_input_standard)
        # print("Sparse Input Grad:\n", grad_input_sparse)
        exit(1)
    else:
        print("\nGradient check PASSED.")

if __name__ == "__main__":
    run_gradient_test()