def CORR_find_not_common(a,b):
    """ find noncommon elements of number vectors a and b """
    return list(set(a).symmetric_difference(b))