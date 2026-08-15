"""Orquestra a reconstrução local Bronze -> Silver -> Gold."""
from src.preprocessing import bronze, silver, gold

if __name__ == "__main__":
    print("=" * 60)
    bronze.run_all()
    print("=" * 60)
    silver.run_all()
    print("=" * 60)
    gold.run_all()
    print("=" * 60)
    print("Pipeline concluída. Tabelas Gold em data/gold/*.parquet")
