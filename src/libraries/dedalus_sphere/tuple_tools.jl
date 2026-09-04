
dual(t) = tuple((-e for e in t)...)

apply(p) = t -> tuple((t[i+1] for i in p)...)

sum_(k) = t -> sum((t[i+1] for i in k if 0 <= i < length(t)), init=0)

remove(k) = t -> tuple((s for (i, s) in enumerate(t) if !(i-1 in k))...)

replace_at(j, n) = t -> tuple((i-1 == j ? n : s for (i, s) in enumerate(t))...)

function tuple2index(tup, indexing)
    digits = [findfirst(==(s), indexing) - 1 for s in tup]
    base = length(indexing)
    result = 0
    for d in digits
        result = result * base + d
    end
    return result
end

function index2tuple(index, rank, indexing)
    base = length(indexing)
    if rank == 0
        return ()
    end
    digits = Int[]
    val = index
    for _ in 1:rank
        pushfirst!(digits, val % base)
        val = val ÷ base
    end
    return tuple((indexing[d+1] for d in digits)...)
end
