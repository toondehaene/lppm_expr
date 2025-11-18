import polars as pl
from lppm_expr import stateful_acc, vertical_scan

df = pl.DataFrame({"values": [1.0, 2.0, 3.0, 4.0, 5.0,]})
result = df.select(stateful_acc(pl.col("values")).alias("accumulated"))
result2 = df.select(vertical_scan(pl.col("values")).alias("vertical_scan"))
print(result2)