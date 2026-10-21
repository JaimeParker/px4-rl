"""tools/gen_ctypes.py: parsing rules, and the installed structs against a fresh generation from the headers."""
import ctypes
import importlib.util
from pathlib import Path

import pytest

import px4rl

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tools" / "gen_ctypes.py"
HEADERS = [ROOT / "include" / "px4rl" / "c_api.h", ROOT / "include" / "px4rl" / "plant.h"]

pytestmark = pytest.mark.skipif(not GENERATOR.is_file(), reason="run from a source tree")


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("gen_ctypes", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(generator, tmp_path, header_text, name="sample.h"):
    header = tmp_path / name
    header.write_text(header_text)
    output = tmp_path / "out.py"
    generator.main(["--output", str(output), str(header)])
    namespace = {}
    exec(compile(output.read_text(), str(output), "exec"), namespace)
    return namespace


def test_parser_handles_comments_macros_nested_structs_and_forward_declarations(generator, tmp_path):
    namespace = generate(
        generator,
        tmp_path,
        """
        #define PX4RL_MAX_ROTORS 4
        #define PX4RL_OUTCOME_OK 0 /* trailing comment */
        struct px4rl_later;                       /* forward declaration, not a definition */
        typedef struct px4rl_inner {
            float a;      /* a comment with { braces } and ; semicolons
                             over two lines */
            uint8_t flag; // another one }
        } px4rl_inner;
        typedef struct px4rl_outer {
            int32_t count;
            px4rl_inner items[PX4RL_MAX_ROTORS];
            float grid[PX4RL_MAX_ROTORS][3];
        } px4rl_outer;
        """,
    )
    inner, outer = namespace["Inner"], namespace["Outer"]
    assert ctypes.sizeof(inner) == 8 and ctypes.sizeof(outer) == 4 + 4 * 8 + 4 * 3 * 4
    assert [name for name, _ in outer._fields_] == ["count", "items", "grid"]
    assert len(outer().items) == 4 and len(outer().grid[0]) == 3
    assert namespace["MAX_ROTORS"] == 4 and namespace["OUTCOME_OK"] == 0
    assert namespace["STRUCTS"] == ("Inner", "Outer")


@pytest.mark.parametrize("member", ["double d;", "unsigned long u;", "px4rl_unknown x;", "float f[N_UNKNOWN];",
                                    "float (*callback)(void);"])
def test_unsupported_members_are_errors_not_guesses(generator, tmp_path, member):
    with pytest.raises(ValueError):
        generate(generator, tmp_path, f"typedef struct px4rl_bad {{ float ok; {member} }} px4rl_bad;")


def test_a_header_without_structs_is_an_error(generator, tmp_path):
    with pytest.raises(ValueError):
        generate(generator, tmp_path, "#define PX4RL_X 1\n")


def test_installed_structs_are_what_the_generator_makes_of_the_headers_now(generator, tmp_path):
    output = tmp_path / "fresh.py"
    generator.main(["--output", str(output), *map(str, HEADERS)])
    installed = Path(px4rl._structs.__file__).read_text()
    assert installed == output.read_text(), "px4rl was built from different headers; rebuild it"
