import numpy as np

def calculate_minf1(Pi, Qi, Ui, Ri, si):
  minf1_terms = (Pi ** 2 + Qi ** 2) / (Ui ** 2) * si * Ri
  minf1_total = np.sum(minf1_terms)
  return minf1_total

def calculate_minf2(Ii, INi):
    """
    minf2 = max(Ii/INi) - min(Ii/INi)
    """

    Ii_pu = Ii / INi
    minf2_pu = np.max(Ii_pu) - np.min(Ii_pu)

    return minf2_pu