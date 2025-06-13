#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste simples para verificar imports
"""

import streamlit as st

def main():
    st.title("🧠 NeuralTrain Forge - Teste")
    st.write("Aplicação funcionando!")
    
    st.success("✅ Imports básicos funcionando")
    
    # Teste de imports específicos
    try:
        import plotly.graph_objects as go
        st.success("✅ Plotly importado com sucesso")
    except ImportError as e:
        st.error(f"❌ Erro ao importar Plotly: {e}")
    
    try:
        import pandas as pd
        st.success("✅ Pandas importado com sucesso")
    except ImportError as e:
        st.error(f"❌ Erro ao importar Pandas: {e}")
    
    try:
        import numpy as np
        st.success("✅ NumPy importado com sucesso")
    except ImportError as e:
        st.error(f"❌ Erro ao importar NumPy: {e}")

if __name__ == "__main__":
    main()

