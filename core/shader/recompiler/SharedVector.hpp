#ifndef CORE_SHADER_RECOMPILER_SHAREDVECTOR_HPP
#define CORE_SHADER_RECOMPILER_SHAREDVECTOR_HPP

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

template <typename T>
class SharedVector {
public:
    using Storage = std::vector<T>;
    using value_type = T;
    using size_type = typename Storage::size_type;
    using const_reference = typename Storage::const_reference;
    using const_iterator = typename Storage::const_iterator;
    using iterator = const_iterator;

    SharedVector() = default;
    SharedVector(Storage values) : holder(values.empty() ? nullptr : std::make_shared<Storage>(std::move(values))) {}
    SharedVector(std::initializer_list<T> values) : SharedVector(Storage(values)) {}
    SharedVector& operator=(Storage values) {
        holder = values.empty() ? nullptr : std::make_shared<Storage>(std::move(values));
        return *this;
    }
    SharedVector& operator=(std::initializer_list<T> values) { return *this = Storage(values); }

    [[nodiscard]] const Storage& Values() const { return holder != nullptr ? *holder : Empty(); }
    operator const Storage&() const { return Values(); }

    [[nodiscard]] size_type size() const { return Values().size(); }
    [[nodiscard]] bool empty() const { return Values().empty(); }
    [[nodiscard]] const_reference operator[](size_type index) const { return Values()[index]; }
    [[nodiscard]] const_reference at(size_type index) const { return Values().at(index); }
    [[nodiscard]] const_reference front() const { return Values().front(); }
    [[nodiscard]] const_reference back() const { return Values().back(); }
    [[nodiscard]] const_iterator begin() const { return Values().begin(); }
    [[nodiscard]] const_iterator end() const { return Values().end(); }
    [[nodiscard]] const_iterator cbegin() const { return Values().begin(); }
    [[nodiscard]] const_iterator cend() const { return Values().end(); }

    [[nodiscard]] decltype(auto) operator[](size_type index) { return Mutable()[index]; }
    [[nodiscard]] decltype(auto) at(size_type index) { return Mutable().at(index); }
    [[nodiscard]] decltype(auto) front() { return Mutable().front(); }
    [[nodiscard]] decltype(auto) back() { return Mutable().back(); }
    void push_back(const T& value) { Mutable().push_back(value); }
    template <typename... Args>
    decltype(auto) emplace_back(Args&&... args) { return Mutable().emplace_back(std::forward<Args>(args)...); }
    void resize(size_type count) { Mutable().resize(count); }
    void resize(size_type count, const T& value) { Mutable().resize(count, value); }
    void assign(size_type count, const T& value) { Mutable().assign(count, value); }
    void reserve(size_type count) { Mutable().reserve(count); }
    void clear() { holder.reset(); }
    [[nodiscard]] Storage& Mutable() {
        if (holder == nullptr || holder.use_count() != 1) holder = std::make_shared<Storage>(Values());
        return *holder;
    }

    void Detach() {
        if (holder != nullptr && holder.use_count() != 1) holder = std::make_shared<Storage>(*holder);
    }
    [[nodiscard]] bool SharesStorageWith(const SharedVector& other) const { return holder != nullptr && holder == other.holder; }

    friend bool operator==(const SharedVector& left, const SharedVector& right) { return left.holder == right.holder || left.Values() == right.Values(); }
    friend bool operator==(const SharedVector& left, const Storage& right) { return left.Values() == right; }

private:
    static const Storage& Empty() {
        static const Storage empty;
        return empty;
    }

    std::shared_ptr<Storage> holder;
};

}

#endif
