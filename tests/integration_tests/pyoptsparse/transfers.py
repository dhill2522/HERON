"""
  Implements transfer functions
"""

def electric_consume(data, meta):
  '''The CashFlow driver function for the electricity market
  '''
  # print('electric_consume:', data, meta)

  # FIXME: Get this working right once the dispatch is in place
  # E = -1 * data['electricity']
  data = {'driver': -1}

  return data, meta

def generator(data, meta):
  effciency = 0.7 # Just a random guess at a turbine efficiency

  if 'steam' in data:
    # Determine the electricity output for a given steam input
    data['electricity'] = effciency * data['steam']
  elif 'electricity' in data:
    # Determine the steam input for a given electricity output
    data['steam'] = -1/effciency * data['electricity']
  else:
    raise Exception("Generator Transfer Function: Neither 'electricity' nor 'steam' given")

  return data, meta


def flex_price(data, meta):
  sine = meta['HERON']['RAVEN_vars']['Signal']
  t = meta['HERON']['time_index']
  # DispatchManager
  # scale electricity consumed to flex between -1 and 1
  amount = - 2 * (sine[t] - 0.5)
  data = {'reference_price': amount}
  return data, meta
