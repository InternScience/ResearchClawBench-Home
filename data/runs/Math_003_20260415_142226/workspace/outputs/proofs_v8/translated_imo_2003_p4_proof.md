# Proof for translated_imo_2003_p4

**Target:** Segment pq ≅ Segment qr

**Status:** ❌ Not proven

## Statistics
- Initial facts: 31
- Derived facts: 22
- Total facts: 53
- Iterations: 3

## Proof Steps

1. **Points d, a, b, c are concyclic**
   - Rule: `rule_1`
   - From: Segment od ≅ Segment oa, Segment oa ≅ Segment ob, Segment ob ≅ Segment oc

2. **Points b1, a, b, c are concyclic**
   - Rule: `rule_1`
   - From: Segment ob1 ≅ Segment oa, Segment oa ≅ Segment ob, Segment ob ≅ Segment oc

3. **Points d1, a, b, c are concyclic**
   - Rule: `rule_1`
   - From: Segment od1 ≅ Segment oa, Segment oa ≅ Segment ob, Segment ob ≅ Segment oc

4. **∠(rd, qd) = ∠(ab, ca)**
   - Rule: `rule_8`
   - From: Line rd ⊥ Line ab, Line qd ⊥ Line ca

5. **∠(rd, pd) = ∠(ab, bc)**
   - Rule: `rule_8`
   - From: Line rd ⊥ Line ab, Line pd ⊥ Line bc

6. **∠(qd, rd) = ∠(ca, ab)**
   - Rule: `rule_8`
   - From: Line qd ⊥ Line ca, Line rd ⊥ Line ab

7. **∠(qd, pd) = ∠(ca, bc)**
   - Rule: `rule_8`
   - From: Line qd ⊥ Line ca, Line pd ⊥ Line bc

8. **∠(pd, rd) = ∠(bc, ab)**
   - Rule: `rule_8`
   - From: Line pd ⊥ Line bc, Line rd ⊥ Line ab

9. **∠(pd, qd) = ∠(bc, ca)**
   - Rule: `rule_8`
   - From: Line pd ⊥ Line bc, Line qd ⊥ Line ca

10. **∠(od, da) = ∠(da, oa)**
   - Rule: `rule_13`
   - From: Segment od ≅ Segment oa

11. **∠(ob1, b1a) = ∠(b1a, oa)**
   - Rule: `rule_13`
   - From: Segment ob1 ≅ Segment oa

12. **∠(oa, ab) = ∠(ab, ob)**
   - Rule: `rule_13`
   - From: Segment oa ≅ Segment ob

13. **∠(b1c, ca) = ∠(ca, b1a)**
   - Rule: `rule_13`
   - From: Segment b1c ≅ Segment b1a

14. **∠(od1, d1a) = ∠(d1a, oa)**
   - Rule: `rule_13`
   - From: Segment od1 ≅ Segment oa

15. **∠(ob, bc) = ∠(bc, oc)**
   - Rule: `rule_13`
   - From: Segment ob ≅ Segment oc

16. **∠(d1c, ca) = ∠(ca, d1a)**
   - Rule: `rule_13`
   - From: Segment d1c ≅ Segment d1a

17. **∠(bd, ba) = ∠(cd, ca)**
   - Rule: `rule_3`
   - From: Points d, a, b, c are concyclic

18. **∠(bb1, ba) = ∠(cb1, ca)**
   - Rule: `rule_3`
   - From: Points b1, a, b, c are concyclic

19. **∠(bd1, ba) = ∠(cd1, ca)**
   - Rule: `rule_3`
   - From: Points d1, a, b, c are concyclic

20. **∠(rd, rd) = ∠(ab, ab)**
   - Rule: `rule_9`
   - From: ∠(rd, qd) = ∠(ab, ca), ∠(qd, rd) = ∠(ca, ab)

21. **∠(qd, qd) = ∠(ca, ca)**
   - Rule: `rule_9`
   - From: ∠(qd, rd) = ∠(ca, ab), ∠(rd, qd) = ∠(ab, ca)

22. **∠(pd, pd) = ∠(bc, bc)**
   - Rule: `rule_9`
   - From: ∠(pd, rd) = ∠(bc, ab), ∠(rd, pd) = ∠(ab, bc)
