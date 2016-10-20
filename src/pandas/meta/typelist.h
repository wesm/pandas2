// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders
#pragma once

namespace pandas {

// This class is a compile time forward linked list of types.
//
// Here is an example of how it can be used
// to run a test case over multiple types:
//
// class Tester
// {
//   public:
//     template <typename T>
//     void operator()()
//     {
//         MyTemplateType<T> type;
//         // Test the type
//     }
// };
//
// Tester tester;
// TypeList<int32_t, int64_t, float, double>().iterate(tester);
//
// This will call the Tester's operator() function for
// each of int32_t, int64_t, float, and double.
//
template <typename CURRENT, typename... ARGS>
class TypeList {
 public:
  // The current type
  using current = CURRENT;

  // The next type
  using next = TypeList<ARGS...>;

  // The length
  constexpr static auto length = 1 + next::length;

  // Whether or not this is the last one in the list.
  // This can be used for writing SFINAE functions
  // that iterate over the types in the type list
  constexpr static bool last = false;

  constexpr TypeList() {}

  template <typename... OTHER_ARGS>
  constexpr TypeList<CURRENT, ARGS..., OTHER_ARGS...> operator+(
      const TypeList<OTHER_ARGS...>& other) const {
    return TypeList<CURRENT, ARGS..., OTHER_ARGS...>();
  }

  // Call the functor for each type in this type list.
  // The functor must have an operator with the signature:
  //
  // template <typename T> operator()()
  //
  template <typename FUNCTOR>
  void Iterate(FUNCTOR& functor) const {
    functor.template operator()<CURRENT>();
    next().Iterate(functor);
  }

  // Same as iterate, just in reverse order
  template <typename FUNCTOR>
  void ReverseIterate(FUNCTOR& functor) const {
    next().ReverseIterate(functor);
    functor.template operator()<CURRENT>();
  }

  template <typename SEARCH, std::size_t OFFSET = 0>
  constexpr std::size_t IndexOf() const {
    return (std::is_same<SEARCH, current>::value ? OFFSET
                                                 : next().IndexOf<SEARCH, OFFSET + 1>());
  }

  template <typename... OTHER_ARGS>
  constexpr auto CartesianProduct(const TypeList<OTHER_ARGS...>& other) -> decltype(
      TypeList<CURRENT>().CartesianProduct(other) + next().CartesianProduct(other)) {
    return TypeList<CURRENT>().CartesianProduct(other) + next().CartesianProduct(other);
  }

  template <std::size_t INDEX>
  class LookupByIndex {
   public:
    // Set the next_index to zero when INDEX == 0. Without this,
    // the next_index would wrap around to std::numeric_limits<std::size_t>::max()
    // and there is a compile failure when at is called for he the last
    // element in the list (the partial specialization below).
    static constexpr auto next_index = (INDEX > 0 ? INDEX - 1 : INDEX);

    using next_type = typename next::template At<next_index>::type;

    using type = typename std::conditional<INDEX == 0, current, next_type>::type;
  };

  template <std::size_t INDEX>
  using At = LookupByIndex<INDEX>;
};

// Specialization of TypeList for a list of length 1.
// This it the terminating node in the forward linked
// list of the variadic TypeList class above
template <typename CURRENT>
class TypeList<CURRENT> {
 public:
  using current = CURRENT;

  constexpr static auto length = 1;

  constexpr static bool last = true;

  template <typename... OTHER_ARGS>
  constexpr TypeList<CURRENT, OTHER_ARGS...> operator+(
      const TypeList<OTHER_ARGS...>& other) const {
    return TypeList<CURRENT, OTHER_ARGS...>();
  }

  template <typename FUNCTOR>
  void Iterate(FUNCTOR& functor) const {
    functor.template operator()<CURRENT>();
  }

  template <typename FUNCTOR>
  void ReverseIterate(FUNCTOR& functor) const {
    functor.template operator()<CURRENT>();
  }

  template <typename SEARCH, std::size_t OFFSET = 0>
  constexpr std::size_t IndexOf() const {
    return (std::is_same<SEARCH, current>::value ? OFFSET : throw std::out_of_range(
                                                                "Cannot find type"));
  }

  template <typename OTHER>
  constexpr auto CartesianProduct(const TypeList<OTHER>& other) const
      -> TypeList<std::tuple<CURRENT, OTHER>> {
    return TypeList<std::tuple<CURRENT, OTHER>>();
  }

  template <typename OTHER, typename... OTHER_ARGS>
  constexpr auto CartesianProduct(const TypeList<OTHER, OTHER_ARGS...>& other) const
      -> decltype(TypeList<std::tuple<CURRENT, OTHER>>() +
                  this->CartesianProduct(TypeList<OTHER_ARGS...>())) {
    return TypeList<std::tuple<CURRENT, OTHER>>() +
           this->CartesianProduct(TypeList<OTHER_ARGS...>());
  }

  template <typename SEARCH, std::size_t INDEX = 0>
  class LookupByType {
   public:
    static constexpr auto value = (std::is_same<SEARCH, current>::value ? 0 : -1);
  };

  template <std::size_t INDEX>
  using At = std::enable_if<INDEX == 0, current>;
};
}
