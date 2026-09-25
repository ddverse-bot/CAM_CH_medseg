
from .busi import make_datasets as make_busi
from .acdc import make_datasets as make_acdc
from .pulmonary import make_datasets as make_pulmonary
from .covid import make_datasets as make_covid

def make_datasets(name,root,**kwargs):
    if name=="busi": return make_busi(root,**kwargs)
    if name=="acdc": return make_acdc(root,**kwargs)
    if name in ("pulmonary","pu2756"): return make_pulmonary(root,**kwargs)
    if name=="covid": return make_covid(root,**kwargs)
    raise ValueError(f"Unknown dataset: {name}")
