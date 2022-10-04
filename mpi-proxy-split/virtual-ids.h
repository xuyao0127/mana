/****************************************************************************
 *   Copyright (C) 2019-2021 by Gene Cooperman, Rohan Garg, Yao Xu          *
 *   gene@ccs.neu.edu, rohgarg@ccs.neu.edu, xu.yao1@northeastern.edu        *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License in the files COPYING and COPYING.LESSER.  If not, see           *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

#pragma once
#ifndef MPI_VIRTUAL_IDS_H
#define MPI_VIRTUAL_IDS_H

#include <mpi.h>
#include <unordered_map>

#include "virtualidtable.h"
#include "jassert.h"
#include "jconvert.h"
#include "split_process.h"
#include "dmtcp.h"

// Convenience macros
#define MpiCommList  dmtcp_mpi::MpiVirtualization<MPI_Comm>
#define MpiGroupList dmtcp_mpi::MpiVirtualization<MPI_Group>
#define MpiTypeList  dmtcp_mpi::MpiVirtualization<MPI_Datatype>
#define MpiOpList    dmtcp_mpi::MpiVirtualization<MPI_Op>
#define MpiFileList  dmtcp_mpi::MpiVirtualization<MPI_File>
#define MpiCommKeyvalList    dmtcp_mpi::MpiVirtualization<int>
#define MpiRequestList    dmtcp_mpi::MpiVirtualization<MPI_Request>
#ifndef NEXT_FUNC
# define NEXT_FUNC(func)                                                       \
  ({                                                                           \
    static __typeof__(&MPI_##func)_real_MPI_## func =                          \
                                                (__typeof__(&MPI_##func)) - 1; \
    if (_real_MPI_ ## func == (__typeof__(&MPI_##func)) - 1) {                 \
      _real_MPI_ ## func = (__typeof__(&MPI_##func))pdlsym(MPI_Fnc_##func);    \
    }                                                                          \
    _real_MPI_ ## func;                                                        \
  })
#endif // ifndef NEXT_FUNC
#define REAL_TO_VIRTUAL_FILE(id) \
  MpiFileList::instance("MpiFile", MPI_FILE_NULL).realToVirtual(id)
#define VIRTUAL_TO_REAL_FILE(id) \
  MpiFileList::instance("MpiFile", MPI_FILE_NULL).virtualToReal(id)
#define ADD_NEW_FILE(id) \
  MpiFileList::instance("MpiFile", MPI_FILE_NULL).onCreate(id)
#define REMOVE_OLD_FILE(id) \
  MpiFileList::instance("MpiFile", MPI_FILE_NULL).onRemove(id)
#define UPDATE_FILE_MAP(v, r) \
  MpiFileList::instance("MpiFile", MPI_FILE_NULL).updateMapping(v, r)

#define REAL_TO_VIRTUAL_COMM(id) \
  MpiCommList::instance("MpiComm", MPI_COMM_NULL).realToVirtual(id)
#define VIRTUAL_TO_REAL_COMM(id) \
  MpiCommList::instance("MpiComm", MPI_COMM_NULL).virtualToReal(id)
#define ADD_NEW_COMM(id) \
  MpiCommList::instance("MpiComm", MPI_COMM_NULL).onCreate(id)
#define REMOVE_OLD_COMM(id) \
  MpiCommList::instance("MpiComm", MPI_COMM_NULL).onRemove(id)
#define UPDATE_COMM_MAP(v, r) \
  MpiCommList::instance("MpiComm", MPI_COMM_NULL).updateMapping(v, r)

#define REAL_TO_VIRTUAL_GROUP(id) \
  MpiGroupList::instance("MpiGroup", MPI_GROUP_NULL).realToVirtual(id)
#define VIRTUAL_TO_REAL_GROUP(id) \
  MpiGroupList::instance("MpiGroup", MPI_GROUP_NULL).virtualToReal(id)
#define ADD_NEW_GROUP(id) \
  MpiGroupList::instance("MpiGroup", MPI_GROUP_NULL).onCreate(id)
#define REMOVE_OLD_GROUP(id) \
  MpiGroupList::instance("MpiGroup", MPI_GROUP_NULL).onRemove(id)
#define UPDATE_GROUP_MAP(v, r) \
  MpiGroupList::instance("MpiGroup", MPI_GROUP_NULL).updateMapping(v, r)

#define REAL_TO_VIRTUAL_TYPE(id) \
  MpiTypeList::instance("MpiType", MPI_DATATYPE_NULL).realToVirtual(id)
#define VIRTUAL_TO_REAL_TYPE(id) \
  MpiTypeList::instance("MpiType", MPI_DATATYPE_NULL).virtualToReal(id)
#define ADD_NEW_TYPE(id) \
  MpiTypeList::instance("MpiType", MPI_DATATYPE_NULL).onCreate(id)
#define REMOVE_OLD_TYPE(id) \
  MpiTypeList::instance("MpiType", MPI_DATATYPE_NULL).onRemove(id)
#define UPDATE_TYPE_MAP(v, r) \
  MpiTypeList::instance("MpiType", MPI_DATATYPE_NULL).updateMapping(v, r)

#define REAL_TO_VIRTUAL_OP(id) \
  MpiOpList::instance("MpiOp", MPI_OP_NULL).realToVirtual(id)
#define VIRTUAL_TO_REAL_OP(id) \
  MpiOpList::instance("MpiOp", MPI_OP_NULL).virtualToReal(id)
#define ADD_NEW_OP(id) \
  MpiOpList::instance("MpiOp", MPI_OP_NULL).onCreate(id)
#define REMOVE_OLD_OP(id) \
  MpiOpList::instance("MpiOp", MPI_OP_NULL).onRemove(id)
#define UPDATE_OP_MAP(v, r) \
  MpiOpList::instance("MpiOp", MPI_OP_NULL).updateMapping(v, r)

#define REAL_TO_VIRTUAL_COMM_KEYVAL(id) \
  MpiOpList::instance("MpiCommKeyval", 0).realToVirtual(id)
#define VIRTUAL_TO_REAL_COMM_KEYVAL(id) \
  MpiOpList::instance("MpiCommKeyval", 0).virtualToReal(id)
#define ADD_NEW_COMM_KEYVAL(id) \
  MpiOpList::instance("MpiCommKeyval", 0).onCreate(id)
#define REMOVE_OLD_COMM_KEYVAL(id) \
  MpiOpList::instance("MpiCommKeyval", 0).onRemove(id)
#define UPDATE_COMM_KEYVAL_MAP(v, r) \
  MpiOpList::instance("MpiCommKeyval", 0).updateMapping(v, r)

#if 1
#define REAL_TO_VIRTUAL_REQUEST(id) \
  MpiRequestList::instance("MpiRequest", MPI_REQUEST_NULL).realToVirtual(id)
#define VIRTUAL_TO_REAL_REQUEST(id) \
  MpiRequestList::instance("MpiRequest", MPI_REQUEST_NULL).virtualToReal(id)
#define ADD_NEW_REQUEST(id) \
  MpiRequestList::instance("MpiRequest", MPI_REQUEST_NULL).onCreate(id)
#define REMOVE_OLD_REQUEST(id) \
  MpiRequestList::instance("MpiRequest", MPI_REQUEST_NULL).onRemove(id)
#define UPDATE_REQUEST_MAP(v, r) \
  MpiRequestList::instance("MpiRequest", MPI_REQUEST_NULL).updateMapping(v, r)
#else
#define VIRTUAL_TO_REAL_REQUEST(id) id
#define ADD_NEW_REQUEST(id) id
#define UPDATE_REQUEST_MAP(v, r) r
#endif

namespace dmtcp_mpi
{

  template<typename T>
  class MpiVirtualization
  {
    public:
#ifdef JALIB_ALLOCATOR
      static void* operator new(size_t nbytes, void* p) { return p; }
      static void* operator new(size_t nbytes) { JALLOC_HELPER_NEW(nbytes); }
      static void  operator delete(void* p) { JALLOC_HELPER_DELETE(p); }
#endif
      static MpiVirtualization& instance(const char *name, T nullId)
      {
	// FIXME:
	// dmtcp_mpi::MpiVirtualization::instance("MpiGroup", 1)
	//                                       ._vIdTable.printMaps(true)
	// to access _virTableMpiGroup in GDB.
	// We need a cleaner way to access it.
	if (strcmp(name, "MpiOp") == 0) {
	  static MpiVirtualization<T> _virTableMpiOp(name, nullId);
	  return _virTableMpiOp;
	} else if (strcmp(name, "MpiComm") == 0) {
	  static MpiVirtualization<T> _virTableMpiComm(name, nullId);
	  return _virTableMpiComm;
	} else if (strcmp(name, "MpiGroup") == 0) {
	  static MpiVirtualization<T> _virTableMpiGroup(name, nullId);
	  return _virTableMpiGroup;
	} else if (strcmp(name, "MpiType") == 0) {
	  static MpiVirtualization<T> _virTableMpiType(name, nullId);
	  return _virTableMpiType;
	} else if (strcmp(name, "MpiCommKeyval") == 0) {
	  static MpiVirtualization _virTableMpiCommKeyval(name, nullId);
	  return _virTableMpiCommKeyval;
	} else if (strcmp(name, "MpiRequest") == 0) {
	  static MpiVirtualization _virTableMpiRequest(name, nullId);
	  return _virTableMpiRequest;
	} else if (strcmp(name, "MpiFile") == 0) {
	  static MpiVirtualization _virTableMpiFile(name, nullId);
	  return _virTableMpiFile;
	}
	JWARNING(false)(name)(nullId).Text("Unhandled type");
	static MpiVirtualization _virTableNoSuchObject(name, nullId);
	return _virTableNoSuchObject;
      }

      T virtualToReal(T virt)
      {
        // Don't need to virtualize the null id
        if (virt == _nullId) {
          return virt;
        }
        // DMTCP virtual id table already does the lock around the table.
        // FIXME: Even with an empty map, we are seeing 1 microsecond overhead.
        return _vIdTable.virtualToReal(virt);
      }

      T realToVirtual(T real)
      {
        // Don't need to virtualize the null id
        if (real == _nullId) {
          return real;
        }
        // DMTCP virtual id table already does the lock around the table.
        return _vIdTable.realToVirtual(real);
      }

      // Adds the given real id to the virtual id table and creates a new
      // corresponding virtual id.
      // Returns the new virtual id on success, null id otherwise
      T onCreate(T real)
      {
        T vId = _nullId;
        // Don't need to virtualize the null id
        if (real == _nullId) {
          return vId;
        }
        // DMTCP virtual id table already does the lock around the table.
        if (_vIdTable.realIdExists(real)) {
          // Adding a existing real id is a legal operation and
          // we should not report warning/error.
          // For example, MPI_Comm_group accesses the group associated with
          // given communicator. It can be called multiple times from
          // different localtions. They should get the same virtual id and
          // real id of the same group.
          // JWARNING(false)(real)(_vIdTable.getTypeStr())
          //         (_vIdTable.realToVirtual(real))
          //         .Text("Real id exists. Will overwrite existing mapping");
          vId = _vIdTable.realToVirtual(real);
        } else {
          if (!_vIdTable.getNewVirtualId(&vId)) {
            JWARNING(false)(real)(_vIdTable.getTypeStr())
              .Text("Failed to create a new vId");
          } else {
            _vIdTable.updateMapping(vId, real);
          }
        }
        return vId;
      }

      // Removes virtual id from table and returns the real id corresponding
      // to the virtual id; if the virtual id does not exist in the table,
      // returns null id.
      T onRemove(T virt)
      {
        T realId = _nullId;
        // Don't need to virtualize the null id
        if (virt == _nullId) {
          return realId;
        }
        // DMTCP virtual id table already does the lock around the table.
        if (_vIdTable.virtualIdExists(virt)) {
          realId = _vIdTable.virtualToReal(virt);
          _vIdTable.erase(virt);
        } else {
          JWARNING(false)(virt)(_vIdTable.getTypeStr())
                  .Text("Cannot delete non-existent virtual id");
        }
        return realId;
      }

      // Updates the mapping for the given virtual id to the given real id.
      // Returns virtual id on success, null-id otherwise
      T updateMapping(T virt, T real)
      {
        // If the virt is the null id, then return it directly.
        // Don't need to virtualize the null id
        if (virt == _nullId) {
          return _nullId;
        }
        // DMTCP virtual id table already does the lock around the table.
        if (!_vIdTable.virtualIdExists(virt)) {
          JWARNING(false)(virt)(real)(_vIdTable.getTypeStr())
                  (_vIdTable.realToVirtual(real))
                  .Text("Cannot update mapping for a non-existent virt. id");
          return _nullId;
        }
        _vIdTable.updateMapping(virt, real);
        return virt;
      }

    private:
      // Pvt. constructor
      MpiVirtualization(const char *name, T nullId)
        : _vIdTable(name, (T)0, (size_t)999999),
          _nullId(nullId)
      {
      }

      // Virtual Ids Table
      dmtcp::VirtualIdTable<T> _vIdTable;
      // Default "NULL" value for id
      T _nullId;
  }; // class MpiId
};  // namespace dmtcp_mpi

#endif // ifndef MPI_VIRTUAL_IDS_H
