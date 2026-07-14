#!/usr/bin/env python3
"""Inspect an Ascend OM (Offline Model) file using pyacl.

Usage:
    python inspect_om_model.py <model.om> [--device-id N]

Shows model inputs, outputs, dynamic shape info, and memory requirements.
Requires CANN toolkit with pyacl installed (import acl).
"""

import argparse
import os
import sys

ACL_DATA_TYPES = {
    -1: "UNDEFINED",
    0: "FLOAT",
    1: "FLOAT16",
    2: "INT8",
    3: "INT32",
    4: "UINT8",
    6: "INT16",
    7: "UINT16",
    8: "UINT32",
    9: "INT64",
    10: "UINT64",
    11: "DOUBLE",
    12: "BOOL",
    13: "STRING",
    16: "COMPLEX64",
    17: "COMPLEX128",
    27: "BF16",
    29: "INT4",
    30: "UINT1",
    33: "COMPLEX32",
    34: "HIFLOAT8",
    35: "FLOAT8_E5M2",
    36: "FLOAT8_E4M3FN",
}

ACL_FORMATS = {
    -1: "UNDEFINED",
    0: "NCHW",
    1: "NHWC",
    2: "ND",
    3: "NC1HWC0",
    4: "FRACTAL_Z",
    12: "NC1HWC0_C04",
    16: "HWCN",
    27: "NDHWC",
    29: "FRACTAL_NZ",
    30: "NCDHW",
    32: "NDC1HWC0",
    33: "FRACTAL_Z_3D",
    35: "NC",
    47: "NCL",
}


def check_ret(ret, msg):
    if ret != 0:
        print(f"[ERROR] {msg} (ret={ret})", file=sys.stderr)
        sys.exit(1)


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_dims(dims_info):
    dim_count = dims_info.get("dimCount", 0)
    dims = dims_info.get("dims", [])[:dim_count] if dim_count else dims_info.get("dims", [])
    name = dims_info.get("name", "")
    dim_str = "[" + ", ".join(str(d) for d in dims) + "]"
    return f"{name}: {dim_str}" if name else dim_str


def unwrap(result):
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[-1], int):
        return list(result[:-1]), result[-1]
    return [result], 0


def inspect_iodesc(model_desc, num, is_input):
    prefix = "input" if is_input else "output"
    label = "Inputs" if is_input else "Outputs"
    print(f"\n[{label}] ({num})")
    if num == 0:
        print("  (none)")
        return
    for i in range(num):
        if is_input:
            name = acl.mdl.get_input_name_by_index(model_desc, i)
            dims, _ = unwrap(acl.mdl.get_input_dims(model_desc, i))
            dtype = acl.mdl.get_input_data_type(model_desc, i)
            fmt = acl.mdl.get_input_format(model_desc, i)
            size = acl.mdl.get_input_size_by_index(model_desc, i)
        else:
            name = acl.mdl.get_output_name_by_index(model_desc, i)
            dims, _ = unwrap(acl.mdl.get_output_dims(model_desc, i))
            dtype = acl.mdl.get_output_data_type(model_desc, i)
            fmt = acl.mdl.get_output_format(model_desc, i)
            size = acl.mdl.get_output_size_by_index(model_desc, i)
        dims = dims[0] if dims else {}
        dtype_str = ACL_DATA_TYPES.get(dtype, f"UNKNOWN({dtype})")
        fmt_str = ACL_FORMATS.get(fmt, f"UNKNOWN({fmt})")
        print(f"  [{i}] name={name}")
        print(f"      dims={format_dims(dims)}")
        print(f"      dtype={dtype_str}, format={fmt_str}, size={format_size(size)}")


def inspect_dynamic(model_desc, num_inputs):
    print("\n[Dynamic Shape Info]")
    has_dynamic = False

    try:
        batch_result, ret = unwrap(acl.mdl.get_dynamic_batch(model_desc))
        if ret == 0:
            batch = batch_result[0] if batch_result else {}
            batch_count = batch.get("batchCount", 0)
            if batch_count > 0:
                has_dynamic = True
                batches = batch.get("batch", [])[:batch_count]
                print(f"  Dynamic batch: {batch_count} gears: {batches}")
    except Exception:
        pass

    for i in range(num_inputs):
        try:
            hw_result, ret = unwrap(acl.mdl.get_dynamic_hw(model_desc, i))
            if ret == 0:
                hw = hw_result[0] if hw_result else {}
                hw_count = hw.get("hwCount", 0)
                if hw_count > 0:
                    has_dynamic = True
                    hws = hw.get("hw", [])[:hw_count]
                    hw_strs = [f"({h[0]}x{h[1]})" for h in hws]
                    print(f"  Dynamic HW (input {i}): {hw_count} gears: {', '.join(hw_strs)}")
        except Exception:
            pass

    try:
        gear_result, ret = unwrap(acl.mdl.get_input_dynamic_gear_count(model_desc, -1))
        if ret == 0:
            gear_count = gear_result[0] if gear_result else 0
            if gear_count > 0:
                has_dynamic = True
                print(f"  Dynamic dims: {gear_count} gears")
                dims_result, ret2 = unwrap(
                    acl.mdl.get_input_dynamic_dims(model_desc, -1, gear_count)
                )
                if ret2 == 0 and dims_result:
                    dims_list = dims_result[0] if isinstance(dims_result[0], list) else dims_result
                    for idx, d in enumerate(dims_list):
                        print(f"    gear {idx}: {format_dims(d)}")
    except Exception:
        pass

    if not has_dynamic:
        print("  (static shape model, no dynamic dims)")


def inspect_cur_output_dims(model_desc, num_outputs):
    print("\n[Current Output Dims]")
    for i in range(num_outputs):
        try:
            result, ret = unwrap(acl.mdl.get_cur_output_dims(model_desc, i))
            if ret == 0 and result:
                print(f"  [{i}] {format_dims(result[0])}")
            else:
                print(f"  [{i}] (unavailable)")
        except Exception:
            print(f"  [{i}] (unavailable)")


def inspect_memory(model_path):
    print("\n[Memory Requirements]")
    try:
        result, ret = unwrap(acl.mdl.query_size(model_path))
        if ret == 0 and len(result) >= 2:
            print(f"  Work memory:   {format_size(result[0])}")
            print(f"  Weight memory: {format_size(result[1])}")
        else:
            print(f"  (query_size failed: ret={ret})")
    except Exception as e:
        print(f"  (query_size unavailable: {e})")


def inspect_model(model_path, device_id=0):
    try:
        import acl
    except ImportError:
        print(
            "[ERROR] pyacl not found. Install CANN toolkit and ensure "
            "LD_LIBRARY_PATH includes the acl Python library path.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    ret = acl.init()
    check_ret(ret, "acl.init() failed")

    ret = acl.rt.set_device(device_id)
    check_ret(ret, f"acl.rt.set_device({device_id}) failed")

    model_desc = None
    model_id = None

    try:
        model_desc = acl.mdl.create_desc()

        ret = acl.mdl.get_desc_from_file(model_desc, model_path)
        if ret != 0:
            print(
                f"[WARN] get_desc_from_file failed (ret={ret}), "
                "falling back to load_from_file + get_desc...",
                file=sys.stderr,
            )
            load_result, load_ret = unwrap(acl.mdl.load_from_file(model_path))
            check_ret(load_ret, "acl.mdl.load_from_file() failed")
            model_id = load_result[0] if load_result else None
            ret = acl.mdl.get_desc(model_desc, model_id)
            check_ret(ret, "acl.mdl.get_desc() failed")

        print("=" * 70)
        print(f" OM Model Inspection: {os.path.basename(model_path)}")
        print(f" Path: {os.path.abspath(model_path)}")
        print(f" File size: {format_size(os.path.getsize(model_path))}")
        print("=" * 70)

        inspect_memory(model_path)

        num_inputs = acl.mdl.get_num_inputs(model_desc)
        num_outputs = acl.mdl.get_num_outputs(model_desc)

        inspect_iodesc(model_desc, num_inputs, is_input=True)
        inspect_iodesc(model_desc, num_outputs, is_input=False)
        inspect_dynamic(model_desc, num_inputs)
        inspect_cur_output_dims(model_desc, num_outputs)

        print("\n" + "=" * 70)
        print(" Inspection complete.")
        print("=" * 70)
    finally:
        if model_desc is not None:
            acl.mdl.destroy_desc(model_desc)
        if model_id is not None:
            acl.mdl.unload(model_id)
        acl.rt.reset_device(device_id)
        acl.finalize()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect an Ascend OM (Offline Model) file using pyacl."
    )
    parser.add_argument("model", help="Path to the .om model file")
    parser.add_argument(
        "--device-id", type=int, default=0, help="NPU device ID (default: 0)"
    )
    args = parser.parse_args()
    inspect_model(args.model, args.device_id)


if __name__ == "__main__":
    main()
