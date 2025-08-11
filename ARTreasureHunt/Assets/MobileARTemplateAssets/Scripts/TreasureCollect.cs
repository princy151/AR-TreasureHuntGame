using UnityEngine;
using UnityEngine.EventSystems;

public class TreasureCollect : MonoBehaviour
{
    public int scoreValue = 1;

    void Update()
    {
        if ((Input.touchCount > 0 && Input.GetTouch(0).phase == UnityEngine.TouchPhase.Began) || Input.GetMouseButtonDown(0))
        {
            Vector2 inputPosition = Input.touchCount > 0 ? Input.GetTouch(0).position : (Vector2)Input.mousePosition;

            if (EventSystem.current != null && EventSystem.current.IsPointerOverGameObject())
                return;

            Ray ray = Camera.main.ScreenPointToRay(inputPosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit))
            {
                if (hit.transform == transform)
                {
                    Debug.Log("Treasure collected!");

                    TreasureSpawnerAndCollector spawner = FindObjectOfType<TreasureSpawnerAndCollector>();
                    if (spawner != null)
                    {
                        spawner.TreasureCollected(gameObject);
                    }

                    ScoreManager.Instance?.AddScore(scoreValue);

                    Destroy(gameObject);
                }
            }
        }
    }
}
