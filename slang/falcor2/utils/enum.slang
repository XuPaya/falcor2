// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Return true if value contains any bit from flag.
[ForceInline]
public bool is_set<T : __EnumType>(T value, T flag)
{
    return value & flag;
}

/// Test flags represented by unsigned integer bitmasks.
[ForceInline]
public bool is_set(uint value, uint flag)
{
    return (value & flag) != 0;
}

/// Toggle flag in value.
public void flip_bit<T : __EnumType>(inout T value, T flag)
{
    value = is_set(value, flag) ? (value & ~flag) : (value | flag);
}
