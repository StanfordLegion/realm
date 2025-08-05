#ifndef FRAG_LIST_H
#define FRAG_LIST_H

#include <cstdint>
#include <map>
#include <queue>
#include <cassert>

namespace Realm {

  /* Minimal fragment list that reorders elements by 16-bit sequence number.
   * Only supports three operations:
   *   insert(seq, T*) – stores element, returns true if one or more elements
   *                     are now in-order and ready to pull.
   *   pull()          – returns oldest in-order element or nullptr.
   * The list allows arbitrary holes but keeps at most max_holes outstanding. */
  template <typename T>
  class FragList {
  public:
    explicit FragList(uint16_t _head_seq_number = 0xFFFF, unsigned _max_holes = 1024)
      : head_seq_number(_head_seq_number)
      , max_holes(_max_holes)
    {}

    bool insert(uint16_t seq_number, T *elem)
    {
      // already delivered
      if(cmp(seq_number, head_seq_number) <= 0) {
        return false;
      }

      // in-order
      if(seq_number == uint16_t(head_seq_number + 1)) {
        ready_queue.push(elem);
        head_seq_number = seq_number;
        pop_contiguous();
        return true;
      }

      if(buffer.size() >= max_holes) {
        assert(0);
        return false;
      }

      auto ok = buffer.emplace(seq_number, elem);
      return ok.second && pop_contiguous();
    }

    T *pull()
    {
      if(ready_queue.empty()) {
        return nullptr;
      }
      T *e = ready_queue.front();
      ready_queue.pop();
      return e;
    }

  private:
    static inline int cmp(uint16_t a, uint16_t b)
    {
      return int(uint16_t(a - b)); // relies on unsigned wrap
    }

    bool pop_contiguous()
    {
      bool any = false;
      while(true) {
        uint16_t next = uint16_t(head_seq_number + 1);
        auto it = buffer.find(next);
        if(it == buffer.end()) {
          break;
        }
        ready_queue.push(it->second);
        buffer.erase(it);

        head_seq_number = next;
        any = true;
      }
      return any;
    }

    uint16_t head_seq_number;
    unsigned max_holes;
    std::map<uint16_t, T *> buffer;
    std::queue<T *> ready_queue;
  };

} // namespace Realm

#endif
