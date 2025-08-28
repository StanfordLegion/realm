#include "realm/realm_c.h"
#include "test_mock.h"
#include "test_common.h"
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <assert.h>
#include <map>
#include <set>
#include <gtest/gtest.h>

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;

  namespace Config {
    extern bool use_fast_reservation_fallback;
  };
}; // namespace Realm

class CFastReservationBaseTest {
protected:
  void initialize(void)
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplWithReservationFreeList>();
    runtime_impl->init(1);
  }

  void finalize(void) { runtime_impl->finalize(); }

protected:
  std::unique_ptr<MockRuntimeImplWithReservationFreeList> runtime_impl{nullptr};
};

// test the failed cases

class CFastReservationFailedTest : public CFastReservationBaseTest,
                                   public ::testing::Test {
protected:
  void SetUp() override { CFastReservationBaseTest::initialize(); }

  void TearDown() override { CFastReservationBaseTest::finalize(); }
};

TEST_F(CFastReservationFailedTest, CreateNullRuntime)
{
  realm_fast_reservation_t fast_reservation;
  realm_status_t status =
      realm_fast_reservation_create(nullptr, REALM_NO_RESERVATION, &fast_reservation);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CFastReservationFailedTest, CreateInvalidFastReservation)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status =
      realm_fast_reservation_create(runtime, REALM_NO_RESERVATION, nullptr);
  EXPECT_EQ(status, REALM_RESERVATION_ERROR_INVALID_RESERVATION);
}

TEST_F(CFastReservationFailedTest, DestroyNullFastReservation)
{
  realm_status_t status = realm_fast_reservation_destroy(nullptr);
  EXPECT_EQ(status, REALM_RESERVATION_ERROR_INVALID_RESERVATION);
}

TEST_F(CFastReservationFailedTest, WrlockNullFastReservation)
{
  realm_event_t event;
  realm_status_t status = realm_fast_reservation_wrlock(nullptr, &event);
  EXPECT_EQ(status, REALM_RESERVATION_ERROR_INVALID_RESERVATION);
}

TEST_F(CFastReservationFailedTest, RdlockNullFastReservation)
{
  realm_event_t event;
  realm_status_t status = realm_fast_reservation_rdlock(nullptr, &event);
  EXPECT_EQ(status, REALM_RESERVATION_ERROR_INVALID_RESERVATION);
}

TEST_F(CFastReservationFailedTest, UnlockNullFastReservation)
{
  realm_status_t status = realm_fast_reservation_unlock(nullptr);
  EXPECT_EQ(status, REALM_RESERVATION_ERROR_INVALID_RESERVATION);
}

// now we test the success case

class CFastReservationParamTest : public CFastReservationBaseTest,
                                  public ::testing::TestWithParam<bool> {
protected:
  void SetUp() override
  {
    CFastReservationBaseTest::initialize();
    create_from_reservation = GetParam();
    if(create_from_reservation) {
      reservation = Reservation::create_reservation();
    }
  }

  void TearDown() override
  {
    if(create_from_reservation) {
      // TODO: use c api once we implement it
      Reservation resv{reservation};
      resv.destroy_reservation();
    }
    CFastReservationBaseTest::finalize();
  }

  realm_reservation_t reservation{REALM_NO_RESERVATION};
  bool create_from_reservation{false};
};

// TODO: need to get rid of runtime_singleton to test this
TEST_P(CFastReservationParamTest, Create)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_fast_reservation_t fast_reservation;
  realm_status_t status =
      realm_fast_reservation_create(runtime, reservation, &fast_reservation);
  EXPECT_EQ(status, REALM_SUCCESS);

  ASSERT_REALM(realm_fast_reservation_destroy(fast_reservation));
}

TEST_P(CFastReservationParamTest, Destroy)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_fast_reservation_t fast_reservation;
  ASSERT_REALM(realm_fast_reservation_create(runtime, reservation, &fast_reservation));

  realm_status_t status = realm_fast_reservation_destroy(fast_reservation);
  EXPECT_EQ(status, REALM_SUCCESS);
}

TEST_P(CFastReservationParamTest, WrlockNoEvent)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_fast_reservation_t fast_reservation;
  ASSERT_REALM(realm_fast_reservation_create(runtime, reservation, &fast_reservation));

  realm_event_t event;
  realm_status_t status = realm_fast_reservation_wrlock(fast_reservation, &event);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(event, Event::NO_EVENT);

  ASSERT_REALM(realm_fast_reservation_destroy(fast_reservation));
}

// TODO: need to get rid of runtime_singleton to test this
TEST_P(CFastReservationParamTest, DISABLED_WrlockWithEvent)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_fast_reservation_t fast_reservation;
  realm_event_t event;
  // fallback has to be enabled to let lock return an event
  Realm::Config::use_fast_reservation_fallback = true;
  ASSERT_REALM(realm_fast_reservation_create(runtime, reservation, &fast_reservation));
  ASSERT_REALM(realm_fast_reservation_wrlock(fast_reservation, &event));

  realm_status_t status = realm_fast_reservation_wrlock(fast_reservation, &event);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(event, event);
  Realm::Config::use_fast_reservation_fallback = false;

  ASSERT_REALM(realm_fast_reservation_destroy(fast_reservation));
}

TEST_P(CFastReservationParamTest, RdlockNoEvent)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_fast_reservation_t fast_reservation;
  ASSERT_REALM(realm_fast_reservation_create(runtime, reservation, &fast_reservation));

  realm_event_t event;
  realm_status_t status = realm_fast_reservation_rdlock(fast_reservation, &event);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(event, Event::NO_EVENT);

  ASSERT_REALM(realm_fast_reservation_destroy(fast_reservation));
}

TEST_P(CFastReservationParamTest, DoubleRdlockNoEvent)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_fast_reservation_t fast_reservation;
  ASSERT_REALM(realm_fast_reservation_create(runtime, reservation, &fast_reservation));
  realm_event_t event;
  ASSERT_REALM(realm_fast_reservation_rdlock(fast_reservation, &event));

  realm_status_t status = realm_fast_reservation_rdlock(fast_reservation, &event);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(event, Event::NO_EVENT);

  ASSERT_REALM(realm_fast_reservation_destroy(fast_reservation));
}

// TODO: need to get rid of runtime_singleton to test this
TEST_P(CFastReservationParamTest, DISABLED_RdlockWithEvent)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_fast_reservation_t fast_reservation;
  realm_event_t event;
  // fallback has to be enabled to let lock return an event
  Realm::Config::use_fast_reservation_fallback = true;
  ASSERT_REALM(realm_fast_reservation_create(runtime, reservation, &fast_reservation));
  // acquire the lock, such that rdlock will return an event
  ASSERT_REALM(realm_fast_reservation_wrlock(fast_reservation, &event));

  ASSERT_REALM(realm_fast_reservation_rdlock(fast_reservation, &event));

  realm_status_t status = realm_fast_reservation_rdlock(fast_reservation, &event);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(event, event);
  Realm::Config::use_fast_reservation_fallback = false;

  ASSERT_REALM(realm_fast_reservation_destroy(fast_reservation));
}

TEST_P(CFastReservationParamTest, UnlockForWRLock)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_fast_reservation_t fast_reservation;
  ASSERT_REALM(realm_fast_reservation_create(runtime, reservation, &fast_reservation));
  realm_event_t event;
  ASSERT_REALM(realm_fast_reservation_wrlock(fast_reservation, &event));

  realm_status_t status = realm_fast_reservation_unlock(fast_reservation);
  EXPECT_EQ(status, REALM_SUCCESS);

  ASSERT_REALM(realm_fast_reservation_destroy(fast_reservation));
}

TEST_P(CFastReservationParamTest, UnlockForRdlock)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_fast_reservation_t fast_reservation;
  ASSERT_REALM(realm_fast_reservation_create(runtime, reservation, &fast_reservation));
  realm_event_t event;
  ASSERT_REALM(realm_fast_reservation_rdlock(fast_reservation, &event));

  realm_status_t status = realm_fast_reservation_unlock(fast_reservation);
  EXPECT_EQ(status, REALM_SUCCESS);

  ASSERT_REALM(realm_fast_reservation_destroy(fast_reservation));
}

// TODO: enable test for create from reservation
INSTANTIATE_TEST_SUITE_P(CreateFromReservationVariants, CFastReservationParamTest,
                         ::testing::Values(false));