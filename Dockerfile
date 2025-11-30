# 1️⃣ Használj megfelelő Python verziót (max 3.11)
FROM python:3.11-slim

# 2️⃣ Munkakönyvtár beállítása
WORKDIR /app

# 3️⃣ Másold be a projektfájlokat (de ne a `venv` mappát!)
COPY . /app

# 4️⃣ Frissítsd a `pip`-et, `setuptools`-t és `wheel`-t
RUN python -m venv venv && /app/venv/bin/pip install --upgrade pip setuptools wheel

# 5️⃣ Telepítsd a PyTorch-ot CUDA támogatással a hivatalos forrásból
RUN /app/venv/bin/pip install torch==2.6.0

# 6️⃣ Telepítsd a többi csomagot
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# 7️⃣ Állítsd be a környezeti változókat, hogy a venv aktív legyen
ENV PATH="/app/venv/bin:$PATH"

# 8️⃣ Indítsd el a programot
CMD ["python", "app.py"]
