
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
# from generic import Generic
# from marginal import marginal
# from custom import Custom
from .pyomo_dispatch import Pyomo
from .pyoptsparse_dispatch import PyOptSparse

known = {
    #'generic': Generic,
    #'marginal': Marginal,
    #'custom': Custom,
    'pyomo': Pyomo,
    'pyoptsparse': PyOptSparse
}

def get_class(typ):
  """
    Returns the requested dispatcher type.
    @ In, typ, str, name of one of the dispatchers
    @ Out, class, object, class object
  """
  return known.get(typ, None)
