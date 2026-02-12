import jax
import jax.numpy as np

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


def test_integration(tmp_path):
    r"""
    Test a giant nonsequential model with as many active modules from the
    library as possible.
    """

    im = 1.0  # Initial magnitude for parameters

    modules = {
        # AffineEigenvaluePMM contains AffineHermitianMatrix and Eigenvalues,
        # and is itself a SequentialModule
        "aepmm": pmm.modules.AffineEigenvaluePMM(
            matrix_size=3,
            num_eig=2,
            which="SA",
            smoothing=1.0,
            init_magnitude=im,
            bias_term=True,
        ),
        # AffineObservablePMM contains AffineHermitianMatrix, Eigenvectors,
        # Bias, and is itself a SequentialModule
        "aopmm": pmm.modules.AffineObservablePMM(
            matrix_size=3,
            num_eig=2,
            which="LA",
            smoothing=1.0,
            affine_bias_matrix=True,
            num_secondaries=1,
            output_size=2,
            centered=True,
            bias_term=True,
            use_expectation_values=True,
            init_magnitude=im,
        ),
        "C": (
            pmm.modules.Comment("Comment 1"),
            pmm.modules.Comment("Comment 2"),
        ),
        "const": pmm.modules.Constant(
            trainable=True,
            shape={"mat": (2, 2), "vec": (2,)},
            real=False,
            name="C1",
            init_magnitude=im,
        ),
        "concat_leaves": [
            pmm.modules.ConcatenateLeaves(),
            pmm.modules.ConcatenateLeaves(),
        ],
        "eigensystem": pmm.modules.Eigensystem(
            num_eig=2,
            which="EA",
        ),
        # matmul is a subclass of Einsum
        "matmul": pmm.modules.MatMul(),
        # LinearNN contains Flatten, MatMul, Bias, and an activation function
        # and is itself a SequentialModule
        "nn": pmm.modules.LinearNN(
            out_features=2,
            bias=True,
            activation=pmm.modules.PReLU(
                single_parameter=False, init_magnitude=im, real=True
            ),
            real=True,
        ),
        "rs": pmm.modules.Reshape(shape=(2, 2)),
        "tf": pmm.modules.TreeFlatten(),
        "tk": pmm.modules.TreeKey(["y", "x"]),
        # this module will be ommitted from the connections
        "omitted": pmm.modules.LinearNN(
            out_features=2,
            bias=True,
            activation=pmm.modules.PReLU(
                single_parameter=False, init_magnitude=im, real=True
            ),
            real=True,
        ),
        # this module will be connected, but unreachable from the input or
        # output
        "superfluous": pmm.modules.LinearNN(
            out_features=2,
            bias=True,
            activation=pmm.modules.ReLU(),
            real=True,
        ),
    }

    # connect all the modules together
    # the input will be a PyTree of {"x": 2D Array, "y": 1D Array, "z": *}
    connections = {
        # all inputs go to TreeKey which extracts "x" and "y" in reverse order
        "input": "tk",
        # input "x" also goes to Constant, where the input structure is ignored
        "input.x": "const.a.0.0.1.2.b",
        # the matrix from Constant is MatMulled with the output of nn
        # with the order being inferred from the dictionary order here
        "const.mat": "matmul",
        "nn": "matmul",
        # the first element of tk goes to nn
        "tk.1": "nn",
        # the zeroth element goes to the first comment
        "tk.0": ["C.0", "aopmm"],
        # the output of aopmm also goes to the first comment
        "aopmm": "C.0",
        # the leaves from the first comment goes to the first concat_leaves
        "C.0": "concat_leaves.0",
        # the output from concat_leaves.0 goes to aepmm and the omitted module
        "concat_leaves.0": ["aepmm", "omitted"],
        # the output of aepmm goes to the second comment
        "aepmm": "C.1",
        # as well as the output of matmul
        "matmul": "C.1",
        # the leaves from the second comment goes to the second concat_leaves
        "C.1": "concat_leaves.1",
        # the output of concat_leaves.1 goes to rs to reshape to 2x2
        "concat_leaves.1": "rs",
        # the output of rs goes to eigensystem to get eigenvalues/vectors of
        # only the symmetric/Hermitian part
        "rs": "eigensystem",
        # finally, the output of eigensystem is tree flattened to a list of
        # [eigenvalues, eigenvectors]
        "eigensystem": "tf",
        # and tf goes to the output
        "tf": "output",
    }

    key = jax.random.key(0)

    xkey, ykey, zkey = jax.random.split(key, 3)

    batch_size = 10
    data = {
        "x": jax.random.normal(xkey, shape=(batch_size, 3, 5)),
        "y": jax.random.normal(ykey, shape=(batch_size, 4)),
        "z": (
            jax.random.normal(zkey, shape=(batch_size, 2, 2)),
            jax.random.normal(zkey, shape=(batch_size, 4)),
        ),
    }

    nsm = pmm.NonSequentialModel(modules, connections, rng=key, separator=".")

    nsm.compile(
        key, pmm.tree_util.get_shapes(data, slice(1, None)), verbose=True
    )

    out = nsm(data)

    # output should be a list of [eigenvalues, eigenvectors]
    assert isinstance(out, list)
    assert len(out) == 2
    eigvals, eigvecs = out
    assert eigvals.shape == (batch_size, 2)
    assert eigvecs.shape == (batch_size, 2, 2)

    # test saving and loading
    save_path = tmp_path / "nsm_test.npz"
    nsm.save(save_path)
    nsm2 = pmm.NonSequentialModel.from_file(save_path)

    assert nsm.execution_order == nsm2.execution_order
    assert nsm.connections == nsm2.connections
    assert nsm.input_shape == nsm2.input_shape
    assert nsm.output_shape == nsm2.output_shape
    assert nsm.separator == nsm2.separator

    out2 = nsm2(data)
    eigvals2, eigvecs2 = out2
    assert np.allclose(eigvals, eigvals2)
    assert np.allclose(eigvecs, eigvecs2)
