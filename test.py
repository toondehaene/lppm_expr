import polars as pl
from lppm_expr import stateful_acc, cum_sum_weighted

df = pl.DataFrame({"values": [1.0, 2.0, 3.0, 4.0, 5.0,]})
result = df.select(stateful_acc(pl.col("values")).alias("accumulated"))
result2 = df.select(cum_sum_weighted(pl.col("values")).alias("cum_sum_weighted"))
print(result2)