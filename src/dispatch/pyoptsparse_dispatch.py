from .Dispatcher import Dispatcher
from .DispatchState import NumpyState
from utils import InputData

import pyoptsparse
import numpy as np
import sys
import os
import time as time_lib
import traceback

try:
  import _utils as hutils
except (ModuleNotFoundError, ImportError):
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  import _utils as hutils

class PyOptSparse(Dispatcher):
  '''
  Dispatch using pyOptSparse optimization package
  '''

  @classmethod
  def get_input_specs(cls):
    specs = InputData.parameterInputFactory('pyoptsparse', ordered=False, baseNode=None)
    return specs

  def __init__(self):
    print('PyOptDispatch:__init__')
    self.name = 'PyOptSparseDispatcher'
    self._window_length = 10  # FIXME: Get this from the input

    # Defined on call to self.dispatch
    self.components = None
    self.case = None

  def read_input(self, specs):
    print('\n\n\n\nPyOptDispatch:read_input - specs: ', specs)

  def __gen_pool_cons(self, res):
    '''A closure for generating a pool constraint for a resource'''

    def pool_cons(dispatch_window: NumpyState):
      '''A resource pool constraint

      Ensures that the net amount of a resource being consumed, produced and
      stored is zero. Inteded to '''
      time = dispatch_window._times
      n = len(time)
      err = np.zeros(n)

      # FIXME: This is an inefficient way of doing this. Find a better way
      cs = [c for c in self.components if res in c.get_resources()]
      for i, t in enumerate(time):
        for c in cs:
          err[i] += dispatch_window.get_activity(c, res, t)

      # FIXME: This simply returns the sum of the errors over time. There
      # are likely much better ways of handling this.
      # Maybe have a constraint for the resource at each point in time?
      # Maybe use the storage components as slack variables?
      return sum(err)

    return pool_cons

  def _build_pool_cons(self):
    '''Build the pool constraints
    Returns a list of `pool_cons` functions, one for each resource.'''

    cons = []
    for res in self.resources:
      # Generate the pool constraint here
      pool_cons = self.__gen_pool_cons(res)
      cons.append(pool_cons)
    return cons

  def determine_dispatch(self, opt_vars, time):
    '''Determine the dispatch from a given set of optimization
    vars by running the transfer functions. Returns a Numpy dispatch
    object
    @ In, opt_vars, dict, holder for all the optimization variables
    @ In, time, list, time horizon to dispatch over
    '''
    # Initialize the dispatch
    dispatch = NumpyState()
    dispatch.initialize(self.components, self.resource_index_map, time)
    # Dispatch the fixed components
    fixed_comps = [c for c in self.components if c.is_dispatchable() == 'fixed']
    for f in fixed_comps:
      resource = f.get_capacity_var()
      capacity = f.get_capacity(self.meta)[0][resource]
      vals = np.ones(len(time)) * capacity
      dispatch.set_activity_vector(f, f.get_capacity_var(), 0, len(time), vals)
    # Dispatch the independent and dependent components using the vars
    disp_comps = [c for c in self.components if c.is_dispatchable() != 'fixed']
    for d in disp_comps:
      inter = d.get_interaction()
      for i in range(len(time)):
        request = {d.get_capacity_var(): opt_vars[d.name][i]}
        if inter.tag == 'stores':
          # FIXME: Determine what the current resource level is
          lvl = 0
          # FIXME: Determine dt
          dt = 1.0
          bal, meta = inter.produce(request, self.meta, self.sources, dispatch, i, dt, lvl)
        else:
          bal, meta = inter.produce(request, self.meta, self.sources, dispatch, i)
        for res, value in bal.items():
          dispatch.set_activity_indexed(d, self.resource_index_map[d][res], i, value)

    return dispatch, meta

  def _dispatch_pool(self):
    # Steps:
    #   1) Assemble all the vars into a vars dict
    #     A set of vars for each dispatchable component including storage elements
    #     include bound constraints
    #   2) Build the pool constraint functions
    #     Should have one constraint function for each pool
    #     Each constraint will be a function of all the vars
    #   3) Assemble the transfer function constraints
    #   4) Set up the objective function as the double integral of the incremental dispatch
    #   5) Assemble the parts for the optimizer function
    #     a) Declare the variables
    #     b) Declare the constraints
    #     c) Declare the objective function
    #     d) set the optimization configuration (IPOPT/SNOPT, CS/FD...)
    #   6) Run the optimization and handle failed/unfeasible runs
    #   7) Set the activities on each of the components and return the result

    print('\n\n\n\nDEBUG: pyOptSparse dispatcher')
    self.resources = hutils.get_all_resources(self.components)
    self.start_time = time_lib.time()

    # The time discretization for the dispatch is provided by the dispatch manager
    t_begin, t_end, t_num = self.get_time_discr()
    time_horizon = np.linspace(t_begin, t_end, t_num)
    self.resource_index_map = self.meta['HERON']['resource_indexer']

    # Step 1) Find the vars: 1 for each component input where dispatch is not fixed
    self.vs = {} # Min/Max tuples of the various input
    for c in self.components:
      if c.is_dispatchable() == 'fixed':
        # Fixed dispatch components do not contribute to the variables
        continue
      else: # Independent and dependent dispatch
        self.vs[c.name] = {}
        cap_res = c.get_capacity_var()
        capacity = c.get_capacity(self.meta)
        # capacity = c.get_capacity(None, None, None, None)
        self.vs[c.name] = [0, capacity[0][cap_res]] # FIXME: get real min capacity
        self.vs[c.name].sort() # The max capacities are often negative

    full_dispatch = NumpyState()
    full_dispatch.initialize(self.components,
                              self.resource_index_map,
                              time_horizon)

    # FIXME: Add constraints to ensure that the windows overlap

    win_start_i = 0
    win_i = 0
    while win_start_i < t_num:
      win_end_i = win_start_i + self._window_length
      if win_end_i > t_num:
        win_end_i = t_num

      win_horizon = time_horizon[win_start_i:win_end_i]
      win_dispatch = self._dispatch_window(win_horizon, win_i)

      for comp in self.components:
        for res in comp.get_resources():
          full_dispatch.set_activity_vector(
            comp, res, win_start_i, win_end_i,
            win_dispatch.get_activity_vector(comp, res, 0, len(win_horizon))
          )

      win_i += 1
      win_start_i = win_end_i - 1 # Need to overlap time windows

    return full_dispatch

    # return self._dispatch_window(time_horizon, 1)

  def _dispatch_window(self, time_window, win_i):
    print('Dispatching window', win_i)

    # Step 2) Build the resource pool constraint functions
    print('Step 2) Build the resource pool constraints', time_lib.time() - self.start_time)
    pool_cons = self._build_pool_cons()

    # Step 3) Assemble the transfer function constraints
    # this is taken into account by "determining the dispatch"

    # Step 4) Set up the objective function as the double integral of the incremental dispatch
    print('Step 4) Assembling the big function', time_lib.time() - self.start_time)
    def obj(stuff):
      # nonlocal meta
      dispatch, self.meta = self.determine_dispatch(stuff, time_window)
      # At this point the dispatch should be fully determined, so assemble the return object
      things = {}
      # Dispatch the components to generate the obj val
      things['objective'] = self._compute_cashflows(self.components, dispatch, time_window, self.meta)
      # Run the resource pool constraints
      things['resource_balance'] = [cons(dispatch) for cons in pool_cons]
      things['window_overlap'] = []
      # FIXME: Nothing is here to verify ramp rates!
      return things, False

    # Step 5) Assemble the parts for the optimizer function
    print('Step 5) Setting up pyOptSparse', time_lib.time() - self.start_time)
    optProb = pyoptsparse.Optimization(self.case.name, obj)
    for comp in self.vs.keys():
      for comp, bounds in self.vs.items():
        # FIXME: will need to find a way of generating the guess values
        optProb.addVarGroup(comp, len(time_window), 'c', value=-1, lower=bounds[0], upper=bounds[1])
    optProb.addConGroup('resource_balance', len(pool_cons), lower=0, upper=0)
    # if win_i != 0:
    #   optProb.addConGroup('window_overlap', len())
    optProb.addObj('objective')

    # Step 6) Run the optimization
    print('Step 6) Running the dispatch optimization', time_lib.time() - self.start_time)
    try:
      opt = pyoptsparse.OPT('IPOPT')
      sol = opt(optProb, sens='CD')
      # print(sol)
      print('Dispatch optimization successful')
    except Exception as err:
      print('Dispatch optimization failed:')
      traceback.print_exc()
      raise err

    # Step 7) Set the activities on each component
    print('\nCompleted dispatch process\n\n\n')

    win_opt_dispatch, self.meta = self.determine_dispatch(sol.xStar, time_window)
    print(f'Optimal dispatch for win {win_i}:', win_opt_dispatch)
    print('\nReturning the results', time_lib.time() - self.start_time)
    return win_opt_dispatch


  def dispatch(self, case, components, sources, meta):
    self.components = components
    self.case = case
    self.sources = sources
    self.meta = meta
    # Will need to override this here if changing without rerunning outer.
    self._window_length = 10
    return self._dispatch_pool()

# Questions:
# - How should I raise exceptions? What is the raven way?
# - Is there a suggested way of multithreading in HERON?
# - Should I set things like meta as class members or pass them everywhere (functional vs oop)?
# - Tried internalParrallel to true for inner and it failed to import TEAL. Any ideas?
# - Best way of getting dispatch window length from user?

# ToDo:
# - Try priming the initial values better
# - Calculate exact derivatives using JAX
#   - Could use extra meta props to accomplish this
# - Scale the obj func inputs and outputs
# - Find a way to recover the optimal dispatch
# - Integrate storage into the dispatch
#   - formulate storage as a slack variable in the resource constraint
# - Determine the analytical solution for a benchmark problem


# Ideas
# - may need to time the obj func call itself
# - Could try linearizing the transfer function?
#   - Probably not a good idea as it would severely limit the functionality of the transfer
#   - Could be done by a researcher beforehand and used in the Pyomo dispatcher
