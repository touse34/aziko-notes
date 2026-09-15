---
title: CUDA 中 int 换成 bool 会减少寄存器吗？
description: CUDA 中 bool、int 与寄存器分配的关系。
---

# CUDA 中 int 换成 bool 会减少寄存器吗？

结论：**不一定。**

虽然在 C++ 语义上 `bool` 只需要表示 `true / false`，但 CUDA GPU 的寄存器分配并不是简单按照变量的数据类型大小进行的。
