## What and why

## How it was tested

- [ ] `ctest` passes (`cmake -S . -B build -DCMAKE_C_FLAGS="-Wall -Wextra -Wpedantic -Werror"`)
- [ ] Backend change: run on real hardware with its comparison test (say which device and vendor tool)
- [ ] Public headers: append only, `struct_size` on new structs, `nm -gU libuairt` lists the same symbols
- [ ] No vendor SDK files committed (`scripts/check_no_vendor_files.sh`)
- [ ] Docs and CHANGELOG updated

## What I did not verify
