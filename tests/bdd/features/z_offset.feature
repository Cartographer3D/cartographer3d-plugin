Feature: Adjust z-offset

  Background:
    Given a probe
    And the probe has scan calibrated

  Rule: Apply offset from baby stepping

    Example: Scan - Nozzle raised 0.4mm
      Given the probe has scan z-offset 2.0
      And I ran G28
      And I have baby stepped the nozzle 0.4mm up
      When I run the Z_OFFSET_APPLY_PROBE macro
      Then it should set scan z-offset to 1.6

    Example: Scan - Nozzle lowered 0.4mm
      Given the probe has scan z-offset 2.0
      And I ran G28
      And I have baby stepped the nozzle 0.4mm down
      When I run the Z_OFFSET_APPLY_PROBE macro
      Then it should set scan z-offset to 2.4

    Example: Touch - Nozzle raised 0.4mm
      Given the probe has touch calibrated
      And the probe has touch z-offset 0.0
      And I ran TOUCH_HOME
      And I have baby stepped the nozzle 0.4mm up
      When I run the Z_OFFSET_APPLY_PROBE macro
      Then it should set touch z-offset to -0.4

    Example: Touch - Nozzle lowered 0.4mm
      Given the probe has touch calibrated
      And the probe has touch z-offset -0.2
      And I ran TOUCH_HOME
      And I have baby stepped the nozzle 0.4mm down
      When I run the Z_OFFSET_APPLY_PROBE macro
      Then it should set touch z-offset to 0

  Rule: Apply offset to latest homing mode

    Example: Touch after scan
      Given the probe has touch calibrated
      And the probe has scan z-offset 2.0
      And the probe has touch z-offset 0.0
      And I ran G28
      And I ran TOUCH_HOME
      And I have baby stepped the nozzle 0.4mm up
      When I run the Z_OFFSET_APPLY_PROBE macro
      Then it should set touch z-offset to -0.4

    Example: Scan after touch
      Given the probe has touch calibrated
      And the probe has scan z-offset 2.0
      And the probe has touch z-offset 0.0
      And I ran TOUCH_HOME
      And I ran G28
      And I have baby stepped the nozzle 0.4mm up
      When I run the Z_OFFSET_APPLY_PROBE macro
      Then it should set scan z-offset to 1.6
