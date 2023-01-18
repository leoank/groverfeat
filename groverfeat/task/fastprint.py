"""Fast grover fingerprints."""

from time import time_ns
from argparse import Namespace
from logging import Logger
from typing import List

import torch
from torch.utils.data import DataLoader

from grover.data import MolCollator
from grover.data import MoleculeDataset
from grover.util.utils import get_data, create_logger, load_fast_checkpoint


def generate_fastprints(args: Namespace, logger: Logger = None) -> List[List[float]]:
    """
    Generate the fingerprints using tensorrt.

    :param logger:
    :param args: Arguments.
    :return: A list of lists of target fingerprints.
    """
    args.dropout = 0.0 # Added by Ersilia
    checkpoint_path = args.checkpoint_paths[0]
    if logger is None:
        logger = create_logger('fingerprints', quiet=False)
    print('Loading data')
    test_data = get_data(path=args.data_path,
                         args=args,
                         use_compound_names=False,
                         max_data_size=float("inf"),
                         skip_invalid_smiles=False)
    test_data = MoleculeDataset(test_data)

    logger.info(f'Total size = {len(test_data):,}')
    logger.info(f'Generating...')
    
    import numpy as np
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner
    
    # Load engine
    tensor_engine = EngineFromBytes(BytesFromPath(args.engine_path))
    model = load_fast_checkpoint(checkpoint_path, cuda=args.cuda, current_args=args, logger=logger)
    model.eval()
    args.bond_drop_rate = 0
    preds = []
    mol_collator = MolCollator(args=args, shared_dict={})
    num_workers = 4
    mol_loader = DataLoader(test_data,
                            batch_size=32,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=mol_collator)
    
    input_names = ["onnx::Cast_0", "onnx::Cast_1", "onnx::Cast_2", "onnx::Cast_3", "onnx::Cast_4", "no_op1", "no_op2", "onnx::Cast_7"]
    input_types = [np.float32, np.float32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32]
    with TrtRunner(tensor_engine) as runner:
        start = time_ns()
        for item in mol_loader:
            _, batch, features_batch, _, _ = item
            feed = dict(zip(input_names, [np.array(x, dtype=input_types[i]) for i,x in enumerate(batch)]))
            feed.pop("no_op1")
            feed.pop("no_op2")
            with torch.no_grad():
                output = runner.infer(feed_dict=feed)
                output = {
                    "atom_from_atom": torch.from_numpy(output["input.527"]), "bond_from_atom": torch.from_numpy(output["input.551"]),
                    "atom_from_bond": torch.from_numpy(output["input.539"]), "bond_from_bond": torch.from_numpy(output["input.563"])
                }
                batch_preds = model(batch, features_batch, output)
                preds.extend(batch_preds.data.cpu().numpy())
        print(f"Time elasped: {(time_ns() - start) / 1000000}")
    return preds