// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

module attributes {llvm.target_triple = "nvptx64-nvidia-cuda"} {
  llvm.func @add_to_buffer(%buffer: !llvm.ptr<1>, %count: i64, %amount: i32)
      attributes {nvvm.kernel} {
    %thread = nvvm.read.ptx.sreg.tid.x : i32
    %block = nvvm.read.ptx.sreg.ctaid.x : i32
    %width = nvvm.read.ptx.sreg.ntid.x : i32
    %thread64 = llvm.zext %thread : i32 to i64
    %block64 = llvm.zext %block : i32 to i64
    %width64 = llvm.zext %width : i32 to i64
    %first = llvm.mul %block64, %width64 : i64
    %index = llvm.add %first, %thread64 : i64
    %active = llvm.icmp "ult" %index, %count : i64
    llvm.cond_br %active, ^add, ^exit
  ^add:
    %slot = llvm.getelementptr %buffer[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %previous = llvm.load %slot : !llvm.ptr<1> -> i32
    %result = llvm.add %previous, %amount : i32
    llvm.store %result, %slot : i32, !llvm.ptr<1>
    llvm.br ^exit
  ^exit:
    llvm.return
  }
}
