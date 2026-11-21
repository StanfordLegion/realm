// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// The argument buffer contains packed Realm instance IDs. Registration supplies
// a local helper in userdata that resolves a ready instance ID to its storage.
llvm.func @seed_value(%args: !llvm.ptr, %arglen: i64,
                      %userdata: !llvm.ptr, %userlen: i64, %processor: i64) {
  %instance = llvm.load %args : !llvm.ptr -> i64
  %address_of = llvm.load %userdata : !llvm.ptr -> !llvm.ptr
  %buffer = llvm.call %address_of(%instance) : !llvm.ptr, (i64) -> !llvm.ptr
  %value = llvm.mlir.constant(21 : i32) : i32
  llvm.store %value, %buffer : i32, !llvm.ptr
  llvm.return
}

llvm.func @double_value(%args: !llvm.ptr, %arglen: i64,
                        %userdata: !llvm.ptr, %userlen: i64, %processor: i64) {
  %input = llvm.load %args : !llvm.ptr -> i64
  %second_arg = llvm.getelementptr %args[1] : (!llvm.ptr) -> !llvm.ptr, i64
  %output = llvm.load %second_arg : !llvm.ptr -> i64
  %address_of = llvm.load %userdata : !llvm.ptr -> !llvm.ptr
  %input_buffer = llvm.call %address_of(%input) : !llvm.ptr, (i64) -> !llvm.ptr
  %output_buffer = llvm.call %address_of(%output) : !llvm.ptr, (i64) -> !llvm.ptr
  %value = llvm.load %input_buffer : !llvm.ptr -> i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %doubled = llvm.mul %value, %two : i32
  llvm.store %doubled, %output_buffer : i32, !llvm.ptr
  llvm.return
}

llvm.func @increment_value(%args: !llvm.ptr, %arglen: i64,
                           %userdata: !llvm.ptr, %userlen: i64, %processor: i64) {
  %instance = llvm.load %args : !llvm.ptr -> i64
  %address_of = llvm.load %userdata : !llvm.ptr -> !llvm.ptr
  %buffer = llvm.call %address_of(%instance) : !llvm.ptr, (i64) -> !llvm.ptr
  %value = llvm.load %buffer : !llvm.ptr -> i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %incremented = llvm.add %value, %one : i32
  llvm.store %incremented, %buffer : i32, !llvm.ptr
  llvm.return
}

llvm.func @sum_values(%args: !llvm.ptr, %arglen: i64,
                      %userdata: !llvm.ptr, %userlen: i64, %processor: i64) {
  %left = llvm.load %args : !llvm.ptr -> i64
  %second = llvm.getelementptr %args[1] : (!llvm.ptr) -> !llvm.ptr, i64
  %right = llvm.load %second : !llvm.ptr -> i64
  %third = llvm.getelementptr %args[2] : (!llvm.ptr) -> !llvm.ptr, i64
  %result = llvm.load %third : !llvm.ptr -> i64
  %address_of = llvm.load %userdata : !llvm.ptr -> !llvm.ptr
  %left_buffer = llvm.call %address_of(%left) : !llvm.ptr, (i64) -> !llvm.ptr
  %right_buffer = llvm.call %address_of(%right) : !llvm.ptr, (i64) -> !llvm.ptr
  %result_buffer = llvm.call %address_of(%result) : !llvm.ptr, (i64) -> !llvm.ptr
  %a = llvm.load %left_buffer : !llvm.ptr -> i32
  %b = llvm.load %right_buffer : !llvm.ptr -> i32
  %sum = llvm.add %a, %b : i32
  llvm.store %sum, %result_buffer : i32, !llvm.ptr
  llvm.return
}
