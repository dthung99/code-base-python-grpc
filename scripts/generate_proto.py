#!/usr/bin/env python3
"""
Generate Python gRPC code from all proto files in proto/ directory
Cross-platform script that works on Windows, Linux, and macOS
"""

import subprocess
import sys
from pathlib import Path


def generate_grpc_code(
    input_dir: Path = Path("proto"),
    output_dir: Path = Path("src/ai_python_services/proto"),
) -> None:
    """Generate gRPC code from all proto files."""
    print("🔧 Generating gRPC code from all proto files...")

    # Get project root directory
    project_root = Path(__file__).parent.parent
    proto_dir = project_root / input_dir
    output_dir = project_root / output_dir

    # Check if proto directory exists
    if not proto_dir.exists():
        print(f"❌ Proto directory not found: {proto_dir}")
        sys.exit(1)

    # Find all .proto files
    proto_files = list(proto_dir.rglob("*.proto"))

    if not proto_files:
        print(f"❌ No .proto files found in: {proto_dir}")
        sys.exit(1)

    print(f"📁 Found {len(proto_files)} proto files:")
    for proto_file in proto_files:
        print(f"   - {proto_file.name}")

    # Create output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created output directory: {output_dir}")

    # Generate code for each proto file
    failed_files = []

    for proto_file in proto_files:
        print(f"🔧 Generating code for: {proto_file.name}")

        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_file),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"   ✅ Generated: {proto_file.stem}_pb2.py")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed: {proto_file.name}")
            print(f"   Error: {e.stderr}")
            failed_files.append(proto_file.name)
        except FileNotFoundError:
            print("❌ grpc_tools not found. Please install it with:")
            print("   uv add grpcio-tools")
            sys.exit(1)

    # Summary
    if failed_files:
        print(f"\n❌ Failed to generate code for: {', '.join(failed_files)}")
        sys.exit(1)
    else:
        print(f"\n✅ Successfully generated code for all {len(proto_files)} proto files!")
        print(f"📁 Generated files are in: {output_dir}")
        print("🎉 Done! You can now implement your gRPC services.")


def fix_grpc_imports(output_dir: Path = Path("src/ai_python_services/proto")) -> None:
    """Fix relative imports in generated gRPC files."""
    print("🔧 Fixing gRPC imports...")

    for grpc_file in output_dir.glob("*_pb2_grpc.py"):
        print(f"   Fixing imports in: {grpc_file.name}")

        # Read the file
        content = grpc_file.read_text(encoding="utf-8")

        # Find the pb2 module name (e.g., "ai_service_pb2")
        pb2_module = grpc_file.name.replace("_grpc.py", "")

        # Replace absolute import with relative import
        old_import = f"import {pb2_module} as"
        new_import = f"from . import {pb2_module} as"

        if old_import in content:
            content = content.replace(old_import, new_import)
            grpc_file.write_text(content, encoding="utf-8")
            print(f"   ✅ Fixed: {grpc_file.name}")
        else:
            print(f"   ℹ️  No import fix needed: {grpc_file.name}")


def generate_proto_stubs(
    input_dir: Path = Path("proto"),
    output_dir: Path = Path("src/ai_python_services/proto"),
) -> None:
    """Generate .pyi stub files based on proto definitions."""
    print("🔧 Generating proto stub files from proto definitions...")

    # Get project root directory
    project_root = Path(__file__).parent.parent
    proto_dir = project_root / input_dir
    output_dir = project_root / output_dir

    if not proto_dir.exists():
        print(f"❌ Proto directory not found: {proto_dir}")
        return

    # Find all .proto files
    proto_files = list(proto_dir.rglob("*.proto"))

    if not proto_files:
        print(f"❌ No .proto files found in: {proto_dir}")
        return

    # Parse each proto file and generate stub
    for proto_file in proto_files:
        print(f"   Parsing proto file: {proto_file.name}")

        # Read proto file content
        proto_content = proto_file.read_text(encoding="utf-8")

        # Extract message classes
        messages = []
        lines = proto_content.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("message "):
                message_name = line.split()[1]
                fields = []

                # Find fields for this message
                j = i + 1
                while j < len(lines) and not lines[j].strip() == "}":
                    field_line = lines[j].strip()
                    if field_line and not field_line.startswith("//"):
                        # Parse field: "string name = 1;"
                        if "=" in field_line:
                            parts = field_line.split("=")[0].strip().split()
                            if len(parts) >= 2:
                                field_type = parts[0]
                                field_name = parts[1]
                                python_type = _convert_type(field_type)
                                fields.append((field_name, python_type))
                    j += 1

                messages.append((message_name, fields))

        # Generate stub file
        stub_file_name = f"{proto_file.stem}_pb2.pyi"
        stub_file = output_dir / stub_file_name

        stub_content = [
            "from google.protobuf.descriptor import FileDescriptor",
            "from google.protobuf.message import Message",
            "",
            "DESCRIPTOR: FileDescriptor",
            "",
        ]

        # Generate class definitions
        for message_name, fields in messages:
            stub_content.extend(
                [
                    f"class {message_name}(Message):",
                ]
            )

            # Add field type hints
            for field_name, field_type in fields:
                stub_content.append(f"    {field_name}: {field_type}")

            # Add __init__ method
            if fields:
                init_params = ", ".join([f"{name}: {type} = ..." for name, type in fields])
                stub_content.extend(
                    [
                        f"    def __init__(self, *, {init_params}) -> None: ...",
                    ]
                )
            else:
                stub_content.append("    def __init__(self) -> None: ...")

            stub_content.extend(
                [
                    "",
                ]
            )

        # Write stub file
        stub_file.write_text("\n".join(stub_content), encoding="utf-8")
        print(f"   ✅ Generated stub: {stub_file.name}")


def _convert_type(proto_type: str) -> str:
    type_mapping = {
        "string": "str",
        "int32": "int",
        "float": "float",
        "bool": "bool",
        # Add more mappings as needed
    }
    return type_mapping.get(proto_type, proto_type)  # Default case, return as is


# region: Old proto init generation
# def generate_proto_init(
#     output_dir: Path = Path("src/ai_python_services/proto"),
# ) -> None:
#     """Generate __init__.py file for proto package with proper imports."""
#     print("🔧 Generating __init__.py for proto package...")

#     # Get project root directory
#     project_root = Path(__file__).parent.parent
#     output_dir = project_root / output_dir

#     if not output_dir.exists():
#         print(f"❌ Proto output directory not found: {output_dir}")
#         return

#     # Find all generated pb2 and pb2_grpc files
#     pb2_files = list(output_dir.glob("*_pb2.py"))
#     grpc_files = list(output_dir.glob("*_pb2_grpc.py"))

#     if not pb2_files and not grpc_files:
#         print("   ℹ️  No generated proto files found")
#         return

#     # Generate __init__.py content
#     init_content = [
#         '"""',
#         "Generated gRPC proto files.",
#         "This file is auto-generated by scripts/generate_proto.py",
#         '"""',
#         "",
#     ]

#     # Collect all module names and sort them
#     all_modules = []
#     for pb2_file in pb2_files:
#         all_modules.append(pb2_file.stem)
#     for grpc_file in grpc_files:
#         all_modules.append(grpc_file.stem)

#     all_modules = sorted(all_modules)

#     # Add single import line with all modules
#     if all_modules:
#         modules_str = ", ".join(all_modules)
#         init_content.append(f"from . import {modules_str}")
#         init_content.append("")

#         # Add __all__ export list
#         init_content.append("__all__ = [")
#         for module in all_modules:
#             init_content.append(f'    "{module}",')
#         init_content.append("]")

#     # Write __init__.py file
#     init_file = output_dir / "__init__.py"
#     init_file.write_text("\n".join(init_content) + "\n", encoding="utf-8")

#     print(f"   ✅ Generated: {init_file}")
#     print(f"   📦 Exported {len(all_modules)} modules")
# endregion


def main():
    generate_grpc_code(Path("proto"), Path("src/ai_python_services/proto"))
    # fix_grpc_imports(Path("src/ai_python_services/proto"))
    generate_proto_stubs(Path("proto"), Path("src/ai_python_services/proto"))
    # generate_proto_init(Path("src/ai_python_services/proto"))


if __name__ == "__main__":
    main()
