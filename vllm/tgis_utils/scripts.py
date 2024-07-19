# The CLI entrypoint to vLLM.
import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from vllm.model_executor.model_loader.weight_utils import convert_bin_to_safetensor_file
from vllm.scripts import registrer_signal_handlers


def tgis_cli(args: argparse.Namespace) -> None:
    registrer_signal_handlers()

    if args.command == "download-weights":
        download_weights(args.model_name, args.revision, args.token,
                         args.extension, args.auto_convert)
    elif args.command == "convert-to-safetensors":
        convert_bin_to_safetensor_file(args.model_name, args.revision)
    elif args.command == "convert-to-fast-tokenizer":
        convert_to_fast_tokenizer(args.model_name, args.revision,
                                  args.output_path)


def download_weights(
    model_name: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    extension: str = ".safetensors",
    auto_convert: bool = True,
) -> None:
    from vllm.tgis_utils import hub

    print(extension)
    meta_exts = [".json", ".py", ".model", ".md"]

    extensions = extension.split(",")

    if len(extensions) == 1 and extensions[0] not in meta_exts:
        extensions.extend(meta_exts)

    files = hub.download_weights(model_name,
                                 extensions,
                                 revision=revision,
                                 auth_token=token)

    if auto_convert and ".safetensors" in extensions:
        if not hub.local_weight_files(hub.get_model_path(model_name, revision),
                                      ".safetensors"):
            if ".bin" not in extensions:
                print(".safetensors weights not found, \
                    downloading pytorch weights to convert...")
                hub.download_weights(model_name,
                                     ".bin",
                                     revision=revision,
                                     auth_token=token)

            print(".safetensors weights not found, \
                    converting from pytorch weights...")
            convert_bin_to_safetensor_file(model_name, revision)
        elif not any(f.endswith(".safetensors") for f in files):
            print(".safetensors weights not found on hub, \
                    but were found locally. Remove them first to re-convert")
    if auto_convert:
        convert_to_fast_tokenizer(model_name, revision)


def convert_to_fast_tokenizer(
    model_name: str,
    revision: Optional[str] = None,
    output_path: Optional[str] = None,
):
    from vllm.tgis_utils import hub

    # Check for existing "tokenizer.json"
    model_path = hub.get_model_path(model_name, revision)

    if os.path.exists(os.path.join(model_path, "tokenizer.json")):
        print(f"Model {model_name} already has a fast tokenizer")
        return

    if output_path is not None:
        if not os.path.isdir(output_path):
            print(f"Output path {output_path} must exist and be a directory")
            return
    else:
        output_path = model_path

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,
                                                           revision=revision)
    tokenizer.save_pretrained(output_path)

    print(f"Saved tokenizer to {output_path}")
