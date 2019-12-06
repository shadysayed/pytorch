#include <torch/csrc/distributed/rpc/py_rref.h>

#include <torch/csrc/distributed/rpc/python_functions.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace torch {
namespace distributed {
namespace rpc {
///////////////////////////  PyRRef  //////////////////////////////////

PyRRef::PyRRef(std::shared_ptr<RRef> rref) : rref_(std::move(rref)) {
  TORCH_CHECK(rref_, "PyRRef must not wrap nullptr");
}

PyRRef::PyRRef(const py::object& value)
    : PyRRef([&value]() {
        auto rref = RRefContext::getInstance().createOwnerRRef(PyObjectType::get());
        py::object copy(value); // increases refcount
        IValue py_ivalue =
          jit::toIValue(std::move(copy), PyObjectType::get());
        rref->setValue(std::move(py_ivalue));
        return rref;
      }()) {}

bool PyRRef::isOwner() const {
  return rref_->isOwner();
}

WorkerInfo PyRRef::owner() const {
  return RRefContext::getInstance().agent()->getWorkerInfo(rref_->owner());
}

py::object PyRRef::toHere() {
  if (rref_->isOwner()) {
    return localValue();
  } else {
    if (rref_->isPyObj()) {
      // UserRRef<py::object>::toHere() calls python_rpc_handler which acquires
      // GIL.
      return py::cast<py::object>(std::static_pointer_cast<UserRRef>(rref_)->toHere().toPyObject());
    } else {
      IValue value =
          std::static_pointer_cast<UserRRef>(rref_)->toHere();

      {
        // acquiring GIL as torch::jit::toPyObject creates new py::object
        // without grabbing the GIL.
        pybind11::gil_scoped_acquire ag;
        return torch::jit::toPyObject(std::move(value));
      }
    }
  }
}

py::object PyRRef::localValue() {
  TORCH_CHECK(
      rref_->isOwner(),
      "Cannot call localValue() on a non-local reference. Call it on ",
      RRefContext::getInstance().getWorkerName());

  if (rref_->isPyObj()) {
    const py::object& value =
        py::cast<py::object>(std::dynamic_pointer_cast<OwnerRRef>(rref_)->getValue().toPyObject());
    PythonRpcHandler::getInstance().handleException(value);
    {
      // acquiring GIL as the return statement construct a new py::object from
      // a const reference.
      pybind11::gil_scoped_acquire ag;
      return value;
    }
  } else {
    auto value =
        std::dynamic_pointer_cast<OwnerRRef>(rref_)->getValue();
    {
      // acquiring GIL as torch::jit::toPyObject creates new py::object without
      // grabbing the GIL.
      pybind11::gil_scoped_acquire ag;
      return torch::jit::toPyObject(std::move(value));
    }
  }
}

std::string PyRRef::str() const {
  std::stringstream ss;
  if (rref_->isOwner()) {
    ss << "OwnerRRef(" << rref_->rrefId() << ")";
  } else {
    ss << "UserRRef(RRefId = " << rref_->rrefId() << ", ForkId = "
       << std::static_pointer_cast<UserRRef>(rref_)->forkId()
       << ")";
  }
  return ss.str();
}

py::tuple PyRRef::pickle() const {
  auto& ctx = RRefContext::getInstance();
  // TODO: use a dispatch table to pickle/unpickle an RRef, and only only
  // install the dispatch table only when there are indeed RPC activities. As
  // a counter example, checkpointing a model with RRefs should not trigger
  // forks to be added as a fork or a child.
  auto rfd = ctx.prepareChildFork(rref_);
  py::tuple rfd_py = rfd.toPyTuple();
  return rfd_py;
}

PyRRef PyRRef::unpickle(const py::tuple& t) {
  auto& ctx = RRefContext::getInstance();
  auto rfd = RRefForkData::fromPyTuple(t.cast<py::tuple>());
  std::shared_ptr<RRef> rref = nullptr;
  rref = ctx.getOrCreateRRef(rfd, PyObjectType::get());

  ctx.notifyOwnerAndParentOfFork(rfd.forkId_, rfd.parent_, rref);
  return PyRRef(std::move(rref));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
